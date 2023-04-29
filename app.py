import gradio as gr
from src.search import Search

search = Search("config.yaml")

with gr.Blocks() as demo:
    gr.Markdown("Search Sound Effect using this demo.")
    with gr.TabItem("Search from Audio File"):
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(value="太鼓", label="SE Title")
                audio_input = gr.Audio(source="upload")
                ratio = gr.Slider(minimum=0, maximum=1, value=1, label="Weight Parameter. 1 means 'use only text'. 0 means 'use only audio'.")
                topk = gr.Dropdown(
                    [5, 10, 20, 30, 40, 50], value="20", label="Top K"
                )
                button = gr.Button("Search")
            with gr.Column(scale=2):
                output = gr.Dataframe()
    with gr.TabItem("Search from Microphone"):
        with gr.Row():
            with gr.Column(scale=1):
                mic_text_input = gr.Textbox(value="太鼓", label="SE Title")
                mic_audio_input = gr.Audio(source="microphone")
                mic_ratio = gr.Slider(minimum=0, maximum=1, value=1, label="Weight Parameter. 1 means 'use only text'. 0 means 'use only audio'.")
                mic_topk = gr.Dropdown(
                    [5, 10, 20, 30, 40, 50], value="20", label="Top K"
                )
                mic_button = gr.Button("Search")
            with gr.Column(scale=2):
                mic_output = gr.Dataframe()

    button.click(
        search.search, inputs=[text_input, audio_input, ratio, topk], outputs=output
    )
    mic_button.click(
        search.search, inputs=[mic_text_input, mic_audio_input, mic_ratio, mic_topk], outputs=mic_output
    )

demo.launch()
