#!/usr/bin/env python3
"""
Example script demonstrating the clean distillation pipeline.
This script shows how to use the pipeline with minimal code.
"""

import torch
from clean_distillation_pipeline import CleanDistillationPipeline, load_dataset_from_csv


def main():
  """Example usage of the clean distillation pipeline."""

  print("🚀 Clean Knowledge Distillation Pipeline Example")
  print("=" * 50)

  # Configuration
  teacher_model_path = "best_teacher_model.pth"
  csv_path = "./data/annotations.csv"  # Adjust path as needed
  data_root = "./data"

  # Check if teacher model exists
  if not torch.load(teacher_model_path, map_location='cpu'):
    print(f"❌ Teacher model not found at: {teacher_model_path}")
    print("Please ensure you have a trained teacher model saved as 'best_teacher_model.pth'")
    return

  try:
    # Load dataset
    print("📊 Loading dataset...")
    df, class_names, class_id_col = load_dataset_from_csv(csv_path, data_root)
    num_classes = len(class_names)

    print(f"✅ Dataset loaded: {len(df)} samples, {num_classes} classes")
    print(f"   Classes: {class_names}")

    # Initialize pipeline
    print("\n🔧 Initializing distillation pipeline...")
    pipeline = CleanDistillationPipeline(teacher_model_path=teacher_model_path,
                                         num_classes=num_classes,
                                         device='cuda' if torch.cuda.is_available() else 'cpu')

    print(f"✅ Pipeline initialized on {pipeline.device}")

    # Create data loaders
    print("\n📦 Creating data loaders...")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = pipeline.create_data_loaders(
      df, data_root=data_root, batch_size=32, test_size=0.2, val_size=0.1, class_id_col=class_id_col)

    print(f"✅ Data loaders created:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")

    # Train student model
    print("\n🎓 Starting distillation training...")
    print("   This will train a lightweight student model using knowledge from the teacher.")
    print("   The loss combines both soft targets (teacher) and hard targets (ground truth).")

    history, best_val_f1 = pipeline.train_student(
      train_loader=train_loader,
      val_loader=val_loader,
      num_epochs=20,  # Reduced for example
      learning_rate=1e-3,
      weight_decay=1e-4)

    print(f"✅ Training completed!")
    print(f"   Best validation F1: {best_val_f1:.4f}")

    # Evaluate models
    print("\n📈 Evaluating models...")
    results = pipeline.evaluate_models(test_loader)

    print(f"✅ Model comparison results:")
    print(f"   Teacher - Accuracy: {results['teacher']['accuracy']:.4f}, F1: {results['teacher']['f1_score']:.4f}")
    print(f"   Student - Accuracy: {results['student']['accuracy']:.4f}, F1: {results['student']['f1_score']:.4f}")
    print(f"   Compression ratio: {results['compression_ratio']:.1f}x")
    print(f"   Performance retention: {results['performance_retention']:.2%}")

    # Calculate efficiency
    teacher_efficiency = results['teacher']['f1_score'] / results['teacher']['parameters'] * 1e6
    student_efficiency = results['student']['f1_score'] / results['student']['parameters'] * 1e6
    efficiency_gain = student_efficiency / teacher_efficiency

    print(f"   Efficiency gain: {efficiency_gain:.1f}x")

    # Save student model
    print("\n💾 Saving student model...")
    student_model_path = "best_student_model.pth"
    pipeline.save_student_model(student_model_path)

    print(f"✅ Student model saved to: {student_model_path}")

    # Show file size comparison
    import os
    teacher_size = os.path.getsize(teacher_model_path) / (1024 * 1024)
    student_size = os.path.getsize(student_model_path) / (1024 * 1024)

    print(f"\n📊 File size comparison:")
    print(f"   Teacher model: {teacher_size:.2f} MB")
    print(f"   Student model: {student_size:.2f} MB")
    print(f"   Size reduction: {(teacher_size - student_size) / teacher_size * 100:.1f}%")

    # Summary
    print(f"\n🎯 Summary:")
    print(f"   ✅ Successfully created a {results['compression_ratio']:.1f}x smaller model")
    print(f"   ✅ Retained {results['performance_retention']:.1%} of teacher performance")
    print(f"   ✅ Achieved {efficiency_gain:.1f}x efficiency improvement")
    print(f"   ✅ Student model ready for production deployment!")

    # Plot training history
    print(f"\n📊 Generating training plots...")
    pipeline.plot_training_history(history)

    print(f"\n🎉 Distillation pipeline completed successfully!")

  except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("Please check your file paths and ensure the dataset CSV exists.")

  except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check your configuration and try again.")


if __name__ == "__main__":
  main()
