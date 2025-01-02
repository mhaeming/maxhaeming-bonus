import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Displaying CNN Activations in Marimo""")
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import altair as alt
    import pandas as pd
    from torchvision.io import decode_image
    from torchvision.models import alexnet, AlexNet_Weights
    from torchvision.transforms import v2
    return AlexNet_Weights, alexnet, alt, decode_image, mo, pd, torch, v2


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
def _():
    activation = {}
    def get_activation(name):
      # the hook signature
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook
    return activation, get_activation


@app.cell
def _(get_activation, model):
    h1 = model.avgpool.register_forward_hook(get_activation("avgpool"))
    return (h1,)


@app.cell
def _():
    return


@app.cell
def _(torch, v2):
    preprocess_image = v2.Compose([
        v2.Resize(size=(64, 64)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return (preprocess_image,)


@app.cell
def _(decode_image, image_path, model, preprocess_image):
    img = preprocess_image(decode_image(image_path))

    model.forward(img.unsqueeze(0))
    return (img,)


@app.cell
def _(pd, torch):
    def tensor_to_df(tensor: torch.Tensor) -> pd.Dataframe:
        data = tensor.squeeze().reshape(256, -1)
        df = pd.DataFrame({str(i): data[i] for i in range(data.size(1))})
        return df
    return (tensor_to_df,)


@app.cell
def _(activation, tensor_to_df):
    df = tensor_to_df(activation["avgpool"])
    df


    return (df,)


@app.cell
def _(alt, df):
    alt.Chart(df).mark_rect().encode(x="1", y="3")
    return


if __name__ == "__main__":
    app.run()
