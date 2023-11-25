import typer
import yaml
import train_llm



app = typer.Typer()


@app.command()
def train(config_path: str):
    config = yaml.full_load(open(config_path))
    train_llm.train(**config)
    

if __name__ == '__main__':
    app()