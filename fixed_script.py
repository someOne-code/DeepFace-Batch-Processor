#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import psutil
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import tensorflow as tf
from deepface import DeepFace
import logging
from threading import Lock
import gc

# Minimal logging to reduce CPU overhead
logging.basicConfig(level=logging.ERROR)  # Only critical errors
logger = logging.getLogger(__name__)

# Suppress ALL unnecessary outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only fatal errors
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that can be CPU intensive
tf.get_logger().setLevel('FATAL')

# Configuration - CPU-optimized settings
IMAGE_FOLDER = "pics"
OUTPUT_FILE = "results_detailed_gender.csv"

# RAM Configuration - THREADING OPTIMIZED
# Since we use threads, the model is loaded ONCE.
# We only need extra RAM for the image processing buffer per worker.
RAM_BASE_OVERHEAD_MB = 4096  # Base cost for Models (RetinaFace + Analysis models) ~4GB
RAM_PER_WORKER_MB = 512      # ~500MB buffer per concurrent image
RAM_SAFETY_BUFFER_MB = 1024  # Keep 1GB free for system stability
MIN_AVAILABLE_RAM_MB = 2048  # Minimum free RAM to start

# Global model cache to prevent reloading
MODEL_CACHE = {}
model_cache_lock = Lock()

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def print_timestamped(msg, level="INFO"):
    if level in ["TIME", "PERF", "RAM"]:  # Added RAM level
        prefix = "⏰" if level == "TIME" else ("📊" if level == "PERF" else "🧠")
        print(f"{prefix} [{get_timestamp()}] {msg}")

def get_ram_info():
    """Get detailed RAM information"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        'total_mb': mem.total / (1024 * 1024),
        'available_mb': mem.available / (1024 * 1024),
        'used_mb': mem.used / (1024 * 1024),
        'percent': mem.percent,
        'free_mb': mem.free / (1024 * 1024),
        'swap_used_mb': swap.used / (1024 * 1024),
        'swap_percent': swap.percent
    }

def calculate_max_workers_by_ram():
    """Calculate maximum safe number of workers based on available RAM - THREADING OPTIMIZED"""
    ram_info = get_ram_info()

    print_timestamped(f"🧠 RAM Analysis:", "RAM")
    print_timestamped(f"   Total RAM: {ram_info['total_mb']:.0f} MB ({ram_info['total_mb']/1024:.1f} GB)", "RAM")
    print_timestamped(f"   Available RAM: {ram_info['available_mb']:.0f} MB ({ram_info['available_mb']/1024:.1f} GB)", "RAM")

    # Check minimum requirements
    if ram_info['available_mb'] < MIN_AVAILABLE_RAM_MB:
        print_timestamped(f"   ❌ ERROR: Insufficient RAM! Need at least {MIN_AVAILABLE_RAM_MB} MB", "RAM")
        return 0

    # Calculate usable RAM (available - safety buffer - base model cost)
    usable_ram_mb = ram_info['available_mb'] - RAM_SAFETY_BUFFER_MB - RAM_BASE_OVERHEAD_MB

    # If negative, it means we might be tight on loading models, but if we have enough "available", we try at least 1.
    if usable_ram_mb < 0:
        usable_ram_mb = 512 # Minimal buffer

    # Calculate max workers based on RAM
    max_workers_by_ram = int(usable_ram_mb / RAM_PER_WORKER_MB)
    if max_workers_by_ram < 1:
        max_workers_by_ram = 1

    print_timestamped(f"   📊 Base Model Overhead: {RAM_BASE_OVERHEAD_MB} MB", "RAM")
    print_timestamped(f"   📊 RAM per worker buffer: {RAM_PER_WORKER_MB} MB", "RAM")
    print_timestamped(f"   ✅ Max workers by RAM: {max_workers_by_ram}", "RAM")

    return max_workers_by_ram

def check_ram_safety_during_processing():
    """Check if RAM is still safe during processing"""
    ram_info = get_ram_info()
    if ram_info['available_mb'] < 512: # Critical low
        print_timestamped(f"   ⚠️ CRITICAL: Low RAM {ram_info['available_mb']:.0f} MB!", "RAM")
        return False
    return True

def configure_tensorflow_ultra_low_cpu(cpu_threads=1):
    """Ultra-conservative TensorFlow configuration for minimal CPU usage"""
    try:
        # Minimal CPU threading - critical for low CPU usage
        tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)

        # Disable expensive optimizations
        tf.config.optimizer.set_jit(False)
        tf.config.optimizer.set_experimental_options({
            'disable_meta_optimizer': True,
            'pin_to_host_optimization': False,
            'implementation_selector': False,
            'disable_model_pruning': True,
        })
    except Exception as e:
        print_timestamped(f"TensorFlow config warning: {e}", "PERF")

def preload_models_fixed():
    """Correctly preload models into shared memory"""
    print_timestamped("🤖 Starting model preloading (Threading Shared Memory)", "TIME")
    start_time = time.time()

    try:
        # Find a real sample image if possible, otherwise dummy
        sample_image = None
        if os.path.exists(IMAGE_FOLDER):
            for img_file in os.listdir(IMAGE_FOLDER):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_image = os.path.join(IMAGE_FOLDER, img_file)
                    break

        if not sample_image:
             # Create a dummy image
            sample_image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Run a dummy analysis with the EXACT parameters used in processing
        # This forces DeepFace to load:
        # 1. RetinaFace (detector)
        # 2. Age, Gender, Race models
        # into the global singleton cache.
        print_timestamped("   Loading RetinaFace + Analysis models...", "TIME")
        DeepFace.analyze(
            img_path=sample_image,
            actions=['age', 'gender', 'race'],
            detector_backend='retinaface',
            enforce_detection=False,
            silent=True
        )

        load_time = time.time() - start_time
        print_timestamped(f"✅ Models preloaded successfully ({load_time:.2f}s)", "TIME")
        return True

    except Exception as e:
        print_timestamped(f"Model preloading failed: {e}", "PERF")
        return False

def process_image_vectorized(img_path, use_enforce_detection=False):
    """Process single image"""
    img_name = os.path.basename(img_path)

    try:
        # Analyze
        analysis_results = DeepFace.analyze(
            img_path,
            actions=['age', 'gender', 'race'],
            enforce_detection=use_enforce_detection,
            detector_backend='retinaface',
            silent=True
        )

        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results]

        results = []
        for face_idx, face in enumerate(analysis_results, 1):
            age = int(np.round(face.get("age", 0)))
            race = face.get("dominant_race", "n/a")

            gender_data = face.get("gender", {})
            if gender_data:
                # Simple max finding
                dominant_gender = max(gender_data, key=gender_data.get)
                gender_probability = gender_data[dominant_gender]
            else:
                dominant_gender = "n/a"
                gender_probability = 0.0

            results.append([
                img_name, face_idx, age,
                dominant_gender, gender_probability, race
            ])

        return results, False

    except Exception:
        return [[img_name, 0, "no_face_detected", "n/a", 0.0, "n/a"]], True

def process_batch_ultra_efficient(batch_info):
    """Batch processing function"""
    image_paths, use_enforce_detection, batch_id = batch_info

    # No need to configure TF per batch in threading model if done globally

    all_results = []

    for img_path in image_paths:
        results, is_error = process_image_vectorized(img_path, use_enforce_detection)
        all_results.extend(results)

    return all_results

def monitor_resources_minimal():
    """Minimal resource monitoring"""
    try:
        process = psutil.Process()
        cpu = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        ram_info = get_ram_info()
        return {
            'cpu': cpu,
            'memory_mb': memory_mb,
            'system_ram_available_mb': ram_info['available_mb'],
            'system_ram_percent': ram_info['percent']
        }
    except:
        return {'cpu': 0, 'memory_mb': 0, 'system_ram_available_mb': 0, 'system_ram_percent': 0}

def main():
    print_timestamped("🚀 Starting Thread-Optimized DeepFace processing", "TIME")
    total_start = time.time()

    # Configure TF Globally
    configure_tensorflow_ultra_low_cpu(cpu_threads=1)

    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ '{IMAGE_FOLDER}' directory not found!")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg'))]

    if not image_files:
        print(f"❌ No images found in '{IMAGE_FOLDER}'!")
        return

    print_timestamped(f"📁 Found {len(image_files)} images")

    # Calculate workers
    max_workers = calculate_max_workers_by_ram()
    if max_workers == 0:
        return
    # Cap workers reasonable for CPU (e.g., 2x cores or 4-8 max)
    available_cores = cpu_count()
    workers = min(max_workers, available_cores * 2, 8)

    print_timestamped(f"💻 Using {workers} workers (Threads)", "TIME")

    # Preload models
    if not preload_models_fixed():
        print_timestamped("⚠️ Model preloading failed, performance may suffer.", "PERF")

    # Batches
    batch_size = 4 # Smaller batch size for threads is fine
    image_paths = [os.path.join(IMAGE_FOLDER, img) for img in image_files]
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batches.append((image_paths[i:i + batch_size], False, f"B{i}"))

    print_timestamped(f"📦 Created {len(batches)} batches", "TIME")

    all_results = []
    completed_batches = 0

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {
            executor.submit(process_batch_ultra_efficient, batch): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch):
            try:
                res = future.result()
                all_results.extend(res)
                completed_batches += 1
                if completed_batches % 5 == 0:
                     print_timestamped(f"Progress: {completed_batches}/{len(batches)}", "PERF")
            except Exception as e:
                print_timestamped(f"Batch failed: {e}", "PERF")

    total_time = time.time() - total_start

    # Write results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "face_#", "age", "dominant_gender", "gender_probability", "dominant_race"])
        writer.writerows(all_results)

    print_timestamped(f"✅ DONE. Processed {len(image_files)} images in {total_time:.2f}s", "TIME")

if __name__ == "__main__":
    main()
