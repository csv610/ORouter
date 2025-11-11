import argparse

def main():
    """
    Main function to get a response from OpenRouter.
    """
    parser = argparse.ArgumentParser(
        description="Query OpenRouter API models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-q", "--question",
        nargs="?",
        default="Hello, who are you?",
        help="The query to send to the model"
    )
    parser.add_argument(
        "-m", "--model",
        help="Specific model to use (random if not specified)"
    )
    parser.add_argument(
        "-s", "--system",
        help="System prompt to use"
    )
    parser.add_argument(
        "-t", "--temp",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)"
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (default: 0.0)"
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (default: 0.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens in response"
    )
    parser.add_argument(
        "-l", "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output including model name"
    )
    
    args = parser.parse_args()
    
    try:
        config = ModelConfig(
            model=args.model,
            system_prompt=args.system,
            temperature=args.temp,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens
        )
        
        client = OpenRouterClient(config=config)
        
        if args.list_models:
            print("Available models:")
            print("\nAliases (short names):")
            for alias, full_name in sorted(client.MODEL_ALIASES.items()):
                print(f"  {alias:12} -> {full_name}")
            print("\nFull names:")
            for model in client.get_models():
                print(f"  - {model}")
            return
        
        response = client.generate_text(
            user_prompt=args.question,
            verbose=args.verbose
        )
        
        print("\n" + "="*50)
        print(response)
        print("="*50)
        
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"API Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
