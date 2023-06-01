import click

# Main CLI (bat)
@click.group()
def main_cli():
    """The CLI tool for WHite-box Adversarial Toolbox (WHAT)."""
    pass

# what model
@click.group()
def model():
    """Manage Deep Learning Models"""
    pass

# what model list
@model.group('list')
def model_list():
    """List supported models"""
    pass

# what model download
@model.command('download')
def model_download():
    """Download pre-trained models"""
    pass

# what model run
@model.group('run')
def model_run():
    """Run supported models"""
    pass

# what attack
@click.group()
def attack():
    """Manage Attacks"""
    pass

# what attack list
@attack.command('list')
def attack_list():
    """List supported Attacks"""
    pass

# what example
@click.group()
def example():  
    """Manage Examples"""
    pass

# what example list
@example.command('list')
def example_list():
    """List examples"""
    pass

# what exmaple run
@example.group('run')
def example_run():
    """Run examples"""
    pass

def main():
    main_cli.add_command(model)
    main_cli.add_command(attack)
    main_cli.add_command(example)

    model.add_command(model_list)
    model.add_command(model_download)
    model.add_command(model_run)

    attack.add_command(attack_list)

    example.add_command(example_list)
    example.add_command(example_run)

    return main_cli()

if __name__ == "__main__":

    main()
