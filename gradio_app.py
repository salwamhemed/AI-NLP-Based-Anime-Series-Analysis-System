import gradio as gr
import pandas as pd
from theme_classification import theme_classifier
from characters_network import CharactersNetworkGenerator, NamedEntityRecognizer
from character_chatbot import CharacterChatbot
import os

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = [theme.strip() for theme in theme_list_str.split(',')]  # Ensure no extra spaces
    themeClassifier = theme_classifier(theme_list)
    output_df = themeClassifier.get_themes_result(subtitles_path, save_path)

    # Remove 'dialogue' from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    
    # Ensure that only valid themes are selected
    output_df = output_df[theme_list]

    # Aggregate scores and prepare output
    output_df = output_df.sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    return output_df

def get_characters_network(subtitles_path, ner_path):
    characters = CharactersNetworkGenerator()
    ner = NamedEntityRecognizer()
    
    # Get NER and character network data
    ner_df = ner.get_ners(subtitles_path, ner_path)
    characters_df = characters.generate_character_network(ner_df)
    html = characters.draw_characters_network(characters_df)

    return html 
def chat_with_gon_chatbot(message, history):
    character_chatbot = CharacterChatbot("AbdullahTarek/Naruto_Llama-3-8B_3",
                                         huggingface_token = os.getenv('huggingface_token')
                                         )

    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output
def main():
    with gr.Blocks() as demo:
        #HXH Theme Classification section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>HXH Theme Classification</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot(x="Theme", y="Score", title="Series Themes", vertical=False, width=500, height=260)
                    with gr.Column():
                        theme_list_input = gr.Textbox(label='Themes')
                        script_path_input = gr.Textbox(label='Script Path')
                        save_path_input = gr.Textbox(label='Save Path')
                        get_themes_button = gr.Button("Get Themes")

                        def update_plot(theme_list, script_path, save_path):
                            output_df = get_themes(theme_list, script_path, save_path)
                            return plot.update(data=output_df)

                        get_themes_button.click(update_plot, inputs=[theme_list_input, script_path_input, save_path_input], outputs=[plot])
        #HXH Characters Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Characters Network</h1>")
                with gr.Row():
                    with gr.Column():
                        hxh_html_output = gr.HTML()
                    with gr.Column():
                        char_script_path_input = gr.Textbox(label='Script Path')
                        ner_save_path_input = gr.Textbox(label='NER Save Path')
                        get_characters_network_button = gr.Button("Get Characters Network Graph")

                        def update_characters_network(script_path, ner_save_path):
                            html = get_characters_network(script_path, ner_save_path)
                            return hxh_html_output.update(value=html)

                        get_characters_network_button.click(update_characters_network, inputs=[char_script_path_input, ner_save_path_input], outputs=[hxh_html_output])
        #Character Chatbot Section 
        with gr.Row():
            with gr.Column():
             gr.HTML("Character Chatbot")
             gr.ChatInterface(chat_with_gon_chatbot)
    demo.launch(share=True)


if __name__ == '__main__':
    main()
