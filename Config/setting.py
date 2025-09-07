import yaml

def GetDatabaseConfig():
    with open("Config/database_config.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data
def GetOrigindataConfig():
    with open("Config/origindata_config.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

if __name__ == "__main__":
    print(GetOrigindataConfig())