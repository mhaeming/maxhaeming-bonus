# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "pandas==2.2.3",
#     "plotly==5.24.1",
#     "torch==2.5.1",
#     "torchvision==0.20.1",
# ]
# ///

import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Displaying CNN Activations in Marimo""")
    return


@app.cell
def _():
    import marimo as mo
    import json
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    from torchvision.io import decode_image
    from torchvision.models import alexnet, AlexNet_Weights
    from torchvision.transforms import v2
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return (
        AlexNet_Weights,
        alexnet,
        decode_image,
        go,
        json,
        make_subplots,
        mo,
        pd,
        plt,
        torch,
        v2,
    )


@app.cell
def _(mo):
    image_selection = mo.ui.dropdown(["french_loaf.jpg", "maltese.jpeg", "snail.jpeg"], value="snail.jpeg")
    image_selection
    return (image_selection,)


@app.cell
def _(image_selection, mo):
    image_path = f"data/{image_selection.value}"

    mo.image(src=image_path)
    return (image_path,)


@app.cell
def _(activations, go, image_selection, labels, make_subplots):
    preds = activations["classification"].flatten()

    top_preds = preds.argsort()[-5:].tolist()
    fig = make_subplots()
    fig.add_trace(go.Bar(x=list(range(len(preds))), y=preds))
    fig.add_trace(go.Scatter(x=top_preds, y=preds[top_preds], text=[labels[str(i)] for i in top_preds], mode="markers+text", textposition="top center"))
    fig.update_layout(title_text=f"Top-5 Class predictions for {image_selection.value}")
    return fig, preds, top_preds


@app.cell
def _(AlexNet_Weights, alexnet):
    # Initialize the AlexNet model with weights, trained on ImageNet
    model = alexnet(weights=AlexNet_Weights.DEFAULT)

    # Set the model into inference mode
    model = model.eval()
    return (model,)


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Adding forward hooks to AlexNet

        To extract the activations of a particular layer, we need to intercept the information flow of the network.
        Forward hooks allow us to do exactly that.
        """
    )
    return


@app.cell
def _(decode_image, get_activation, image_path, model, preprocess_image):
    activations = {}

    model.avgpool.register_forward_hook(get_activation("avgpool", activations))
    model.features[2].register_forward_hook(get_activation("max_pool1", activations))
    model.features[5].register_forward_hook(get_activation("max_pool2", activations))
    model.classifier[6].register_forward_hook(get_activation("classification", activations))

    img = preprocess_image(decode_image(image_path))

    _ = model.forward(img.unsqueeze(0))
    return activations, img


@app.cell
def _(activations, image_selection, plt):
    def plot_activations(activations):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        
        ax[0].imshow(activations["max_pool1"].squeeze().reshape(64, 729))
        ax[0].set_title("Max Pooling 1")
        ax[1].imshow(activations["max_pool2"].squeeze().reshape(192, 169))
        ax[1].set_title("Max Pooling 2")
        ax[2].imshow(activations["avgpool"].squeeze().reshape(256, 36))
        ax[2].set_title("Average Pooling")
        
        fig.suptitle(f"Layer Activations for {image_selection.value}")

        plt.show()

    plot_activations(activations)
    return (plot_activations,)


@app.cell
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def _(AlexNet_Weights, json, v2):
    preprocess_image = v2.Compose([
        AlexNet_Weights.IMAGENET1K_V1.transforms()
    ])

    def get_activation(name: str, activations):
      # the hook signature
      def hook(model, input, output):
        activations[name] = output.detach()
      return hook

    def get_imagenet_labels(path: str = "data/imagenet_labels.json") -> dict[int, str]:
        with open(path) as f:
            return json.load(f)

    labels = get_imagenet_labels()
    return get_activation, get_imagenet_labels, labels, preprocess_image


if __name__ == "__main__":
    app.run()
