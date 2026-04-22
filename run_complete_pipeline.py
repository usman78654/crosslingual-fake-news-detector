"""
Master script to run the complete 6-week project pipeline
"""

import subprocess
import sys
import time
from datetime import datetime

def run_script(script_name, week_name):
    """Run a Python script and handle errors"""
    print("\n" + "="*80)
    print(f"🚀 Starting: {week_name}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {week_name} completed successfully!")
        print(f"⏱️  Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error in {week_name}")
        print(f"⏱️  Time before error: {elapsed_time:.2f} seconds")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in {week_name}: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("CROSS-LINGUAL FAKE NEWS DETECTION")
    print("Complete 6-Week Project Pipeline")
    print("="*80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run all project phases sequentially:")
    print("  Week 1: Data Exploration")
    print("  Week 2: Data Preprocessing & Baseline")
    print("  Week 3: English Model Training")
    print("  Week 4: Cross-Lingual Evaluation")
    print("  Week 5: Joint Training & Fine-Tuning")
    print("  Week 6: Final Evaluation & Documentation")
    
    input("\nPress Enter to start the complete pipeline...")
    
    project_start_time = time.time()
    
    # Pipeline definition
    pipeline = [
        ("01_data_exploration.py", "Week 1: Data Exploration"),
        ("02_data_preprocessing.py", "Week 2: Data Preprocessing & Baseline"),
        ("03_train_english_model.py", "Week 3: English Model Training"),
        ("04_cross_lingual_eval.py", "Week 4: Cross-Lingual Evaluation"),
        ("05_joint_training_finetuning.py", "Week 5: Joint Training & Fine-Tuning"),
        ("06_final_evaluation.py", "Week 6: Final Evaluation & Documentation")
    ]
    
    results = []
    
    # Run each phase
    for script, week in pipeline:
        success = run_script(script, week)
        results.append((week, success))
        
        if not success:
            print(f"\n⚠️  Pipeline stopped at {week}")
            print("Please fix the error and rerun from this point.")
            break
        
        time.sleep(2)  # Brief pause between phases
    
    # Final summary
    total_time = time.time() - project_start_time
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    for week, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} | {week}")
    
    successful = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} phases")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful == total:
        print("\n🎉 🎉 🎉 PROJECT COMPLETED SUCCESSFULLY! 🎉 🎉 🎉")
        print("\nAll deliverables are ready in the 'results/' directory")
        print("Models are saved in the 'models/' directory")
    else:
        print("\n⚠️  Pipeline incomplete. Please review errors above.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
