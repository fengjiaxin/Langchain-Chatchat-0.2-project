from typing import (
    TYPE_CHECKING,
    Any,
    Tuple
)
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tiktoken


class MinxChatOpenAI:

    @staticmethod
    def import_tiktoken() -> Any:
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_token_ids. "
                "Please install it with `pip install tiktoken`."
            )
        return tiktoken

    @staticmethod
    def get_encoding_model(self) -> Tuple[str, "tiktoken.Encoding"]:
        tiktoken_ = MinxChatOpenAI.import_tiktoken()
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
        try:
            encoding = tiktoken_.encoding_for_model(model)
        except Exception as e:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding
