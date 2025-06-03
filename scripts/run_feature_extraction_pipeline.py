import argparse

def main(args):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature extraction pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for extracted features")
    
    args = parser.parse_args()
    
    main(args)