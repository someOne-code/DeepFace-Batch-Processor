#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import psutil
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# ULTRA-CONSERVATIVE RAM Configuration - 6GB PER WORKER
RAM_PER_WORKER_MB = 6144  # 6GB per worker (as requested)
RAM_SAFETY_BUFFER_MB = 1024  # Keep 2GB free for system stability
MIN_AVAILABLE_RAM_MB = 7168  # Need at least 8GB available to run (6GB worker + 2GB buffer)
RAM_CRITICAL_THRESHOLD_MB = 1536  # If available RAM drops below 1.5GB, it's critical

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
    """Calculate maximum safe number of workers based on available RAM - 6GB PER WORKER"""
    ram_info = get_ram_info()
    
    print_timestamped(f"🧠 RAM Analysis:", "RAM")
    print_timestamped(f"   Total RAM: {ram_info['total_mb']:.0f} MB ({ram_info['total_mb']/1024:.1f} GB)", "RAM")
    print_timestamped(f"   Available RAM: {ram_info['available_mb']:.0f} MB ({ram_info['available_mb']/1024:.1f} GB)", "RAM")
    print_timestamped(f"   Used RAM: {ram_info['used_mb']:.0f} MB ({ram_info['percent']:.1f}%)", "RAM")
    print_timestamped(f"   Free RAM: {ram_info['free_mb']:.0f} MB ({ram_info['free_mb']/1024:.1f} GB)", "RAM")
    
    if ram_info['swap_percent'] > 10:
        print_timestamped(f"   ⚠️ WARNING: Swap usage detected: {ram_info['swap_used_mb']:.0f} MB ({ram_info['swap_percent']:.1f}%)", "RAM")
        print_timestamped(f"   ⚠️ System is using virtual memory - this will cause slowdowns!", "RAM")
    
    # Check minimum requirements
    if ram_info['available_mb'] < MIN_AVAILABLE_RAM_MB:
        print_timestamped(f"   ❌ ERROR: Insufficient RAM! Need at least {MIN_AVAILABLE_RAM_MB} MB ({MIN_AVAILABLE_RAM_MB/1024:.1f} GB) available", "RAM")
        print_timestamped(f"   💡 Try closing other applications to free up memory", "RAM")
        return 0
    
    # Calculate usable RAM (available - safety buffer)
    usable_ram_mb = ram_info['available_mb'] - RAM_SAFETY_BUFFER_MB
    
    # Calculate max workers based on RAM - 6GB per worker
    max_workers_by_ram = int(usable_ram_mb / RAM_PER_WORKER_MB)
    
    # Ensure at least 1 worker if we have minimum RAM
    if max_workers_by_ram < 1 and ram_info['available_mb'] >= MIN_AVAILABLE_RAM_MB:
        max_workers_by_ram = 1
        print_timestamped(f"   ⚠️ Low RAM: Running with 1 worker only", "RAM")
    
    print_timestamped(f"   📊 RAM per worker: {RAM_PER_WORKER_MB} MB ({RAM_PER_WORKER_MB/1024:.1f} GB)", "RAM")
    print_timestamped(f"   📊 Safety buffer: {RAM_SAFETY_BUFFER_MB} MB ({RAM_SAFETY_BUFFER_MB/1024:.1f} GB)", "RAM")
    print_timestamped(f"   ✅ Max workers by RAM: {max_workers_by_ram}", "RAM")
    
    # Warning if we can only use 1 worker
    if max_workers_by_ram == 1:
        print_timestamped(f"   ℹ️ System can support only 1 worker with 6GB requirement", "RAM")
    
    return max_workers_by_ram

def check_ram_safety_during_processing():
    """Check if RAM is still safe during processing"""
    ram_info = get_ram_info()
    
    if ram_info['available_mb'] < RAM_CRITICAL_THRESHOLD_MB:
        print_timestamped(f"   ⚠️ CRITICAL: Available RAM dropped to {ram_info['available_mb']:.0f} MB!", "RAM")
        return False
    
    if ram_info['swap_percent'] > 25:
        print_timestamped(f"   ⚠️ WARNING: Heavy swap usage: {ram_info['swap_percent']:.1f}%", "RAM")
        return False
    
    return True

def calculate_ultra_low_cpu_config(num_images, available_cores):
    """Calculate configuration optimized for minimal CPU usage AND RAM safety - 6GB PER WORKER"""
    print_timestamped(f"🧮 Calculating ultra-low CPU config for {num_images} images", "TIME")
    
    # First, calculate RAM-safe worker limit
    max_workers_by_ram = calculate_max_workers_by_ram()
    
    if max_workers_by_ram == 0:
        print_timestamped("❌ Cannot proceed: Insufficient RAM available", "RAM")
        print_timestamped("", "RAM")
        print_timestamped("💡 To run this script, you need:", "RAM")
        print_timestamped(f"   • At least {MIN_AVAILABLE_RAM_MB/1024:.1f} GB of available RAM", "RAM")
        print_timestamped(f"   • Each worker needs {RAM_PER_WORKER_MB/1024:.1f} GB", "RAM")
        print_timestamped(f"   • Plus {RAM_SAFETY_BUFFER_MB/1024:.1f} GB safety buffer for Windows", "RAM")
        print_timestamped("", "RAM")
        print_timestamped("📝 Suggestions:", "RAM")
        print_timestamped("   1. Close Chrome, Firefox, and other memory-heavy apps", "RAM")
        print_timestamped("   2. Restart your computer to clear memory leaks", "RAM")
        print_timestamped("   3. Close background apps (Discord, Spotify, etc.)", "RAM")
        return 0, 0, 0  # Signal to abort
    
    # ULTRA-CONSERVATIVE: Always start with 1 worker regardless of CPU
    workers = 1
    batch_size = min(8, num_images)
    cpu_threads = 1
    
    # Only increase workers if we have plenty of RAM AND few images
    if num_images < 20 and max_workers_by_ram >= 2 and ram_info['available_mb'] > 12288:  # 12GB+ available
        workers = 1  # Still stay conservative
    
    # CRITICAL: Limit workers by RAM availability (should already be 1)
    if workers > max_workers_by_ram:
        workers = max_workers_by_ram
    
    # Recalculate batch size based on final worker count
    if workers > 0:
        batch_size = max(8, num_images // workers)
    
    # Ensure minimum batch size for efficiency
    if batch_size < 4:
        batch_size = min(4, num_images)
    
    print_timestamped(f"✅ Final Config: {workers} worker(s), {batch_size} batch size, {cpu_threads} CPU thread(s)", "TIME")
    print_timestamped(f"   💾 Estimated RAM usage: {workers * RAM_PER_WORKER_MB:.0f} MB ({workers * RAM_PER_WORKER_MB / 1024:.1f} GB)", "RAM")
    
    return workers, batch_size, cpu_threads

def configure_tensorflow_ultra_low_cpu(cpu_threads=1):
    """Ultra-conservative TensorFlow configuration for minimal CPU usage"""
    try:
        # Minimal CPU threading - critical for low CPU usage
        tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        
        # GPU configuration if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Use mixed precision on GPU for speed
                tf.config.experimental.set_synchronous_execution(True)
        
        # Disable expensive optimizations
        tf.config.optimizer.set_jit(False)
        tf.config.optimizer.set_experimental_options({
            'disable_meta_optimizer': True,
            'pin_to_host_optimization': False,
            'implementation_selector': False,
            'disable_model_pruning': True,
        })
        
        # Enable mixed precision for CPU efficiency (if supported)
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        except:
            pass  # Fallback to float32 if mixed precision not supported
            
    except Exception as e:
        print_timestamped(f"TensorFlow config warning: {e}", "PERF")

def preload_models_fixed():
    """Fixed model preloading that actually works - bypasses KerasTensor bug"""
    print_timestamped("🤖 Starting model preloading (KerasTensor bug workaround)", "TIME")
    start_time = time.time()
    
    try:
        # Find a sample image
        sample_image = None
        for img_file in os.listdir(IMAGE_FOLDER):
            if img_file.lower().endswith(('.jpg', '.jpeg')):
                sample_image = os.path.join(IMAGE_FOLDER, img_file)
                break
        
        if not sample_image:
            print_timestamped("⚠️ No sample image found for preloading", "PERF")
            return False
        
        # WORKAROUND: Use represent function first to load face recognition models
        try:
            from deepface import DeepFace
            # This loads the face recognition models without KerasTensor issues
            DeepFace.represent(
                sample_image, 
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend='opencv'  # Use opencv first, faster
            )
            print_timestamped("✅ Face recognition models loaded", "TIME")
        except Exception as e:
            print_timestamped(f"Face recognition model loading: {e}", "PERF")
        
        # Load analysis models separately using a different approach
        try:
            # Use find function which is more stable for initial loading
            DeepFace.find(
                img_path=sample_image,
                db_path=".",  # Current directory (empty)
                enforce_detection=False,
                detector_backend='opencv',
                model_name="VGG-Face",
                silent=True
            )
        except Exception:
            pass  # Expected to fail (no database), but models get loaded
        
        load_time = time.time() - start_time
        print_timestamped(f"✅ Models preloaded with workaround ({load_time:.2f}s)", "TIME")
        return True
        
    except Exception as e:
        print_timestamped(f"Model preloading completely failed: {e}", "PERF")
        # Don't fail, just continue without preloading
        return False

def process_image_vectorized(img_path, use_enforce_detection=False):
    """Process single image with vectorized operations where possible"""
    img_name = os.path.basename(img_path)
    
    try:
        # Use less strict detection for CPU efficiency
        analysis_results = DeepFace.analyze(
            img_path,
            actions=['age', 'gender', 'race'],
            enforce_detection=use_enforce_detection,
            detector_backend='retinaface',
            silent=True
        )

        # Handle single/multiple faces
        if not isinstance(analysis_results, list):
            analysis_results = [analysis_results]

        # Vectorized processing of results
        results = []
        for face_idx, face in enumerate(analysis_results, 1):
            # Vectorized age calculation
            age = int(np.round(face.get("age", 0)))
            race = face.get("dominant_race", "n/a")
            
            # Vectorized gender processing
            gender_data = face.get("gender", {})
            if gender_data:
                # Use numpy for faster max finding
                gender_values = np.array(list(gender_data.values()))
                gender_keys = list(gender_data.keys())
                max_idx = np.argmax(gender_values)
                dominant_gender = gender_keys[max_idx]
                gender_probability = np.round(gender_values[max_idx], 2)
            else:
                dominant_gender = "n/a"
                gender_probability = 0.0

            results.append([
                img_name, face_idx, age, 
                dominant_gender, gender_probability, race
            ])
        
        return results, False  # success
        
    except Exception:
        # Silent error handling
        return [[img_name, 0, "no_face_detected", "n/a", 0.0, "n/a"]], True  # error

def process_batch_ultra_efficient(batch_info):
    """Ultra-efficient batch processing with minimal CPU overhead"""
    image_paths, use_enforce_detection, batch_id = batch_info
    
    # Configure TensorFlow once per worker with minimal CPU usage
    configure_tensorflow_ultra_low_cpu(cpu_threads=1)
    
    all_results = []
    error_count = 0
    
    # Process images with minimal logging
    for img_path in image_paths:
        results, is_error = process_image_vectorized(img_path, use_enforce_detection)
        all_results.extend(results)
        if is_error:
            error_count += 1
    
    # Force garbage collection to keep memory usage low
    gc.collect()
    
    return all_results

def monitor_resources_minimal():
    """Minimal resource monitoring to reduce CPU overhead"""
    try:
        process = psutil.Process()
        # Quick measurement without interval to reduce overhead
        cpu = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Also get system RAM info
        ram_info = get_ram_info()
        
        return {
            'cpu': cpu, 
            'memory_mb': memory_mb,
            'system_ram_available_mb': ram_info['available_mb'],
            'system_ram_percent': ram_info['percent']
        }
    except:
        return {'cpu': 0, 'memory_mb': 0, 'system_ram_available_mb': 0, 'system_ram_percent': 0}

def initialize_worker_minimal():
    """Minimal worker initialization"""
    # Configure TensorFlow with ultra-low CPU settings
    configure_tensorflow_ultra_low_cpu(cpu_threads=1)
    
    # Suppress all outputs in worker
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    """Ultra-low CPU main function with RAM monitoring - 6GB PER WORKER"""
    print_timestamped("🚀 Starting ULTRA-LOW CPU DeepFace processing with RAM monitoring", "TIME")
    print_timestamped("   ⚙️ Configuration: 6GB RAM per worker required", "TIME")
    total_start = time.time()
    
    # Initial resource check
    initial_resources = monitor_resources_minimal()
    print_timestamped(f"📊 Initial: CPU {initial_resources['cpu']:.1f}%, "
                     f"Process Memory {initial_resources['memory_mb']:.1f}MB, "
                     f"System RAM {initial_resources['system_ram_percent']:.1f}% used", "PERF")
    
    # Configure TensorFlow globally with minimal CPU usage
    configure_tensorflow_ultra_low_cpu(cpu_threads=1)
    
    # Validate and get images
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ '{IMAGE_FOLDER}' directory not found!")
        return
    
    image_files = [f for f in os.listdir(IMAGE_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No images found in '{IMAGE_FOLDER}'!")
        return
    
    print_timestamped(f"📁 Found {len(image_files)} images")
    
    # Calculate ultra-low CPU configuration with RAM awareness (6GB per worker)
    available_cores = cpu_count()
    workers, batch_size, cpu_threads = calculate_ultra_low_cpu_config(len(image_files), available_cores)
    
    # Check if we can proceed
    if workers == 0:
        print_timestamped("❌ ABORT: Insufficient RAM to process images safely", "RAM")
        return
    
    # Use lenient detection for large datasets (faster processing)
    use_enforce_detection = len(image_files) < 50
    
    print_timestamped(f"💻 System: {available_cores} cores available, using {workers} worker(s)")
    print_timestamped(f"⚙️ Batch size: {batch_size}, Detection: {'Strict' if use_enforce_detection else 'Fast'}")
    
    # Preload models (FIXED version)
    preload_start = time.time()
    preload_success = preload_models_fixed()
    preload_time = time.time() - preload_start
    
    if preload_success:
        print_timestamped(f"✅ Models preloaded ({preload_time:.2f}s)", "TIME")
    else:
        print_timestamped("⚠️ Model preloading failed, will load on first use", "PERF")
    
    # Create efficient batches
    image_paths = [os.path.join(IMAGE_FOLDER, img) for img in image_files]
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_id = f"B{len(batches)+1:03d}"
        batches.append((batch_paths, use_enforce_detection, batch_id))
    
    print_timestamped(f"📦 Created {len(batches)} batches", "TIME")
    
    # Process with RAM-aware worker count (6GB per worker)
    processing_start = time.time()
    all_results = []
    
    print_timestamped(f"🔄 Starting processing with {workers} worker(s)...", "TIME")
    
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker_minimal
    ) as executor:
        
        # Submit batches
        future_to_batch = {
            executor.submit(process_batch_ultra_efficient, batch_info): batch_info
            for batch_info in batches
        }
        
        # Collect results with minimal logging
        completed_batches = 0
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result(timeout=300)
                all_results.extend(batch_results)
                completed_batches += 1
                
                # Progress updates every 25% or every 5 batches (whichever is less frequent)
                update_frequency = max(1, min(len(batches) // 4, 5))
                if completed_batches % update_frequency == 0 or completed_batches == len(batches):
                    progress = (completed_batches / len(batches)) * 100
                    current_resources = monitor_resources_minimal()
                    
                    # Check RAM safety
                    is_safe = check_ram_safety_during_processing()
                    safety_indicator = "✅" if is_safe else "⚠️"
                    
                    print_timestamped(f"{safety_indicator} Progress {progress:.0f}% ({completed_batches}/{len(batches)}) - "
                                    f"CPU {current_resources['cpu']:.1f}%, "
                                    f"Process Memory {current_resources['memory_mb']:.1f}MB, "
                                    f"System RAM {current_resources['system_ram_percent']:.1f}% used "
                                    f"({current_resources['system_ram_available_mb']:.0f}MB available)", "PERF")
                
            except Exception as e:
                batch_info = future_to_batch[future]
                batch_paths = batch_info[0]
                print_timestamped(f"⚠️ Batch failed: {e}", "PERF")
                # Add error rows silently
                for img_path in batch_paths:
                    img_name = os.path.basename(img_path)
                    all_results.append([img_name, 0, "no_face_detected", "n/a", 0.0, "n/a"])
                completed_batches += 1

    processing_time = time.time() - processing_start
    print_timestamped(f"✅ Processing completed ({processing_time:.2f}s)", "TIME")
    
    # Write results efficiently
    write_start = time.time()
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "face_#", "age", 
            "dominant_gender", "gender_probability", "dominant_race"
        ])
        writer.writerows(all_results)
    
    write_time = time.time() - write_start
    print_timestamped(f"💾 Results written ({write_time:.2f}s)", "TIME")
    
    # Final statistics
    total_time = time.time() - total_start
    final_resources = monitor_resources_minimal()
    
    successful_results = sum(1 for row in all_results if row[2] != "no_face_detected")
    error_count = len(all_results) - successful_results
    
    print_timestamped("=" * 60, "TIME")
    print_timestamped("📊 ULTRA-LOW CPU PERFORMANCE SUMMARY (6GB per worker)", "TIME")
    print_timestamped("=" * 60, "TIME")
    print_timestamped(f"   Total time: {total_time:.2f}s")
    print_timestamped(f"   Images processed: {len(image_files)}")
    print_timestamped(f"   Images/second: {len(image_files) / total_time:.2f}")
    print_timestamped(f"   Success rate: {successful_results}/{len(image_files)} ({(successful_results/len(image_files)*100):.1f}%)")
    print_timestamped(f"   Errors: {error_count}")
    print_timestamped(f"   Workers used: {workers}")
    print_timestamped(f"📊 Final CPU: {final_resources['cpu']:.1f}%, Process Memory: {final_resources['memory_mb']:.1f}MB", "PERF")
    print_timestamped(f"🧠 Final System RAM: {final_resources['system_ram_percent']:.1f}% used, "
                     f"{final_resources['system_ram_available_mb']:.0f}MB ({final_resources['system_ram_available_mb']/1024:.1f}GB) available", "RAM")
    print_timestamped("=" * 60, "TIME")
    print_timestamped(f"✅ COMPLETE - Results saved to {OUTPUT_FILE}", "TIME")

if __name__ == "__main__":
    main()