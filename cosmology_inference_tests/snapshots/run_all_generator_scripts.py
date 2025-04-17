import os
import glob
import importlib.util
import argparse

def run_all_generators(force=False):
    snapshot_dir = os.path.dirname(__file__)
    pattern = os.path.join(snapshot_dir, "generate_*.py")
    generator_files = sorted(glob.glob(pattern))

    generator_files = [f for f in generator_files if not f.endswith("run_all_generator_scripts.py")]

    print(f"Found {len(generator_files)} generator scripts:")

    for generator_path in generator_files:
        base = os.path.basename(generator_path)
        print(f" - {base}")

    for generator_path in generator_files:
        print(f"\nProcessing {os.path.basename(generator_path)}...")
        spec = importlib.util.spec_from_file_location("generator_module", generator_path)
        generator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator_module)

        generated_files = getattr(generator_module, "GENERATED_SNAPSHOTS", None)
        if generated_files is None:
            print(f"⚠️  WARNING: {os.path.basename(generator_path)} has no GENERATED_SNAPSHOTS list. Skipping.")
            continue

        if not force:
            missing = [f for f in generated_files if not os.path.exists(os.path.join(snapshot_dir, f))]
            if not missing:
                print(f"✅ Skipping {os.path.basename(generator_path)} (all snapshots exist)")
                continue
            else:
                print(f"🔄 Missing files detected, regenerating: {missing}")

        if hasattr(generator_module, "generate_snapshots"):
            generator_module.generate_snapshots()
        else:
            print(f"⚠️  WARNING: {os.path.basename(generator_path)} missing generate_snapshots() function.")

    print("\n✅ Snapshot generation process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all snapshot generators.")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if snapshots exist.")
    args = parser.parse_args()

    run_all_generators(force=args.force)

