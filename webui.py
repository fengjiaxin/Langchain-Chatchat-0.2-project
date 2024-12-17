import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page
import sys
from server.utils import api_address


api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        }
    }

    with st.sidebar:
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api=api)
