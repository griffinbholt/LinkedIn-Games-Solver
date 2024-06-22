import argparse
import os

from pinpoint import PinpointGame
from queens import QueensGame

def main():
    parser = argparse.ArgumentParser(
        prog="linkedin_game.py",
        description="Solves a designated LinkedIn Game (Pinpoint, Queens, or Crossclimb)"
    )
    parser.add_argument("game", choices=["pinpoint", "queens", "crossclimb"])
    parser.add_argument("-i", "--input_file", help="The filepath to the screenshot of the Queens board")
    parser.add_argument("-o", "--output_file", help="The filepath to where the Queens solution will be saved")
    parser.add_argument("-n", "--num_letters", help="The number of letters in each word in Crossclimb")
    args = parser.parse_args()

    if args.game == "pinpoint":
        # Command-line requirements for Pinpoint
        assert args.input_file is None, "-i/--input_file is only applicable for the game Queens"
        assert args.output_file is None, "-o/--output_file is only applicable for the game Queens"
        assert args.num_letters is None, "-n/--num_letters is only applicable for the game Crossclimb"
        
        # OpenAI API Key Requirement
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "You must have an OpenAI API Key. Set the environment variable OPENAI_API_KEY to the key value."

        # Run the Pinpoint game solver
        pinpoint_game = PinpointGame(api_key=api_key)
        pinpoint_game.run_openai_method1()
    elif args.game == "queens":
        # Command-line requirements for Queens
        assert args.input_file is not None, "You must provide a filepath to a screenshot of the Queens board"
        assert args.output_file is not None, "You must provide a filepath to where the Queens solution will be saved"
        assert args.num_letters is None, "-n/--num_letters is not applicable for the game Pinpoint"

        # Run the Queens game solver
        queens_game = QueensGame(screenshot_path=args.input_file)
        queens_game.solve(save_to_path=args.output_file)
    elif args.game == "crossclimb":
        # Command-line requirements for Crossclimb
        assert args.input_file is None, "-i/--input_file is only applicable for the game Queens"
        assert args.output_file is None, "-o/--output_file is only applicable for the game Queens"
        assert args.num_letters is not None, "You must provide the number of letters in each word in Crossclimb"

        # OpenAI API Key Requirement
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "You must have an OpenAI API Key. Set the environment variable OPENAI_API_KEY to the key value."

        print("Crossclimb is not yet implemented.")  # TODO

if __name__ == "__main__":
    main()
