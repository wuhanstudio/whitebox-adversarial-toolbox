import click

what_attack_list = [
    ('TOG', 'Object Detection', 'Adversarial Objectness Gradient Attacks in Real-time Object Detection Systems.'),
    ('PCB', 'Object Detection', 'A Man-in-the-Middle Hardware Attack against Object Detection.')
]

# Main CLI (what)
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
    max_len = max([len(x[0]) for x in what_attack_list])
    for i, attack in enumerate(what_attack_list, start=1):
        print('{} : {:<{w}s}\t{}'.format(i, attack[0], attack[1], w=max_len))

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
