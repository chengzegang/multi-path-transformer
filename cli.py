import typer
import yaml
import train_llm

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(config_path: str):
    import torch
    torch._dynamo.reset()
    config = yaml.full_load(open(config_path))
    train_llm.train(**config)


if __name__ == "__main__":
    
    app()
