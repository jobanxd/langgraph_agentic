import yaml
from pathlib import Path

class AgentProfile:
    # Getting YAML Filepath
    BASE_DIR = Path(__file__).parent
    CONFIG_FILE = BASE_DIR / 'agentprofiles.yaml'

    def __init__(self, agent_name: str):
        if not agent_name:
            raise ValueError("Agent name is required.")
        
        with open(self.CONFIG_FILE, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        if not config:
            raise ValueError("Agent config file not found.")
        
        # Find the agent profile
        profile = None
        for profile in config['agent_profiles']:
            if profile['agent_name'] == agent_name:
                agent_profile = profile
                break

        if not agent_profile:
            raise ValueError(f"Agent '{agent_name}' not found!")
        
        # Set properties directly from YAML
        self.name = agent_profile['agent_name']
        self.model_id = agent_profile['model_id']

        # Load filepath for description and instruction
        description_filepath = self.BASE_DIR / 'profiles' / agent_profile['description_filepath']
        instructions_filepath = self.BASE_DIR / 'profiles' / agent_profile['instructions_filepath']

        # Load description from file
        with open(description_filepath, 'r', encoding='utf-8') as file:
            self.description = file.read()

        # Load instruction from file
        with open(instructions_filepath, 'r', encoding='utf-8') as file:
            self.instruction = file.read()

if __name__ == "__main__":
    root_agent_profile = AgentProfile(agent_name='root_agent')
    print(root_agent_profile.description)