from openai import OpenAI


class PinpointGame():
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.messages = [
            {"role": "system", "content": "We are going to play a game called Pinpoint. All 5 clues belong to a common category. Guess the category."}
        ]

    def run_openai_method1(self):
        previous_result = ""
        for i in range(5):
            clue = input("Enter clue: ")
            self.messages += [
                {"role": "user", "content": previous_result + "Clue #" + str(i + 1) + ": " + clue + "\nWhat is your category guess?\n"}
            ]
            chat_completion = self.client.chat.completions.create(messages=self.messages, model="gpt-3.5-turbo")
            reply = chat_completion.choices[0].message.content
            self.messages += [
                {"role": "assistant", "content": reply}
            ]
            print("Category guess:", reply)

            accepted_input = False
            while not accepted_input:
                result = input("Correct? (y/n): ")
                if result.lower() == "y":
                    print("Congratulations! You won!")
                    return
                elif result.lower() == "n":
                    accepted_input = True
                else:
                    print("Invalid input. Please enter y/n.")
            previous_result = "Incorrect. "
        print("Sorry, you did not guess the correct category in 5 attempts.")

    def run_openai_method2(self):
        previous_result = ""
        clues_so_far = ""
        for i in range(5):
            clue = input("Enter clue: ")
            if len(clues_so_far) != 0:
                clues_so_far += ", "
            clues_so_far += clue
            self.messages += [
                {"role": "user", "content": previous_result + "Clues: [" + clues_so_far + "]\nWhat is your category guess?\n"}
            ]
            chat_completion = self.client.chat.completions.create(messages=self.messages, model="gpt-3.5-turbo")
            reply = chat_completion.choices[0].message.content
            self.messages += [
                {"role": "assistant", "content": reply}
            ]
            print("Category guess:", reply)

            accepted_input = False
            while not accepted_input:
                result = input("Correct? (y/n): ")
                if result.lower() == "y":
                    print("Congratulations! You won!")
                    return
                elif result.lower() == "n":
                    accepted_input = True
                else:
                    print("Invalid input. Please enter y/n.")
            previous_result = "Incorrect. "
        print("Sorry, you did not guess the correct category in 5 attempts.")
