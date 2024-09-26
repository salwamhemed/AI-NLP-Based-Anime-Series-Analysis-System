import gradio as gr
import pandas as pd
from theme_classification import theme_classifier
from characters_network import CharactersNetworkGenerator, NamedEntityRecognizer

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    themeClassifier = theme_classifier(theme_list)
    output_df = themeClassifier.get_themes_result(subtitles_path, save_path)

    # Remove 'dialogue' from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    
    # Ensure that only valid themes are selected
    output_df = output_df[theme_list]

    output_df = output_df.sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    return output_df

def get_characters_network(subtitles_path, ner_path):
    characters = CharactersNetworkGenerator()
    ner = NamedEntityRecognizer()
    
    ner_df = ner.get_ners(subtitles_path, ner_path)
    characters_df = characters.generate_character_network(ner_df)
    html = characters.draw_characters_network(characters_df)

    return html 


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>HXH Theme Classification</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot(x="Theme", y="Score", title="Series Themes", vertical=False, width=500, height=260)
                    with gr.Column():
                        theme_list = gr.Textbox(label='Themes')
                        data_path = gr.Textbox(label='Script path')
                        save_path = gr.Textbox(label='Save path')
                        get_themes_button = gr.Button("Get Themes")

                        def update_plot(theme_list, data_path, save_path):
                            output_df = get_themes(theme_list, data_path, save_path)
                            return gr.BarPlot.update(data=output_df)

                        get_themes_button.click(update_plot, inputs=[theme_list, data_path, save_path], outputs=[plot])
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Characters Network </h1>")
                with gr.Row():
                    with gr.Column():
                        hxh_html = gr.HTML()
                    with gr.Column():
                        data_path = gr.Textbox(label='Script path')
                        Ner_save_path = gr.Textbox(label='Ners Save Path')
                        get_characters_network_button = gr.Button("Get Characters Network Graph")
                        get_characters_network_button.click(get_characters_network, inputs=[data_path, Ner_save_path], outputs=[hxh_html])

                        

    demo.launch(share=True)

if __name__ == '__main__':
    main()
