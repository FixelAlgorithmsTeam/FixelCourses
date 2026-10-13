# Python STD
from enum import auto, Enum, unique
# import math

from dataclasses import dataclass
import base64
import io
import re

# Data
import numpy as np

# Machine Learning

# Deep Learning

# Image Processing / Computer Vision
from PIL import Image

# Optimization

# Python STD

# Auxiliary

# Visualization

# Miscellaneous
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

# Typing
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Self, Set, Tuple, Union
from numpy.typing import NDArray


TIME_REGEX: Final[re.Pattern[str]] = re.compile(r"^\d{2}:\d{2}$")

# Models:
# - OpenAI (https://developers.openai.com/api/docs/models)
#   - gpt-6.1-sol
#   - gpt-6-luna

def _EncodeImageToDataUri( mImg: NDArray ) -> str:
        # `mImg` is expected to be a HWC uint8 NumPy array. Convert it to a PNG data URI.
        pilImg = Image.fromarray(mImg)

        buffer = io.BytesIO()
        pilImg.save(buffer, format = 'PNG')
        encodedImg = base64.b64encode(buffer.getvalue()).decode('ascii')

        return f'data:image/png;base64,{encodedImg}'

def _EncodeImageToBytes( mImg: NDArray ) -> bytes:
        # `mImg` is expected to be a HWC uint8 NumPy array. Convert it to a PNG byte array.
        pilImg = Image.fromarray(mImg)

        buffer = io.BytesIO()
        pilImg.save(buffer, format = 'PNG')
        
        return buffer.getvalue()

@unique
class ModelSupplier(Enum):
    GEMINI = auto()
    OPENAI = auto()


class LLMSession():
    def __init__( self, apiKey: str, modelName: str, /, *, systemMsg: Optional[str] = None ) -> None:

        self.modelName    = modelName
        self.systemMsg    = systemMsg
        self.oModelClient = OpenAI(api_key = apiKey)
        self.lMessages: List[ChatCompletionMessageParam] = []

        if systemMsg is not None:
            self.lMessages.append({'role': 'system', 'content': systemMsg})

    def Prompt( self, promptMsg: str, mImg: Optional[NDArray[np.uint8]] = None ) -> str:

        dMsgUser: ChatCompletionUserMessageParam = {'role': 'user', 'content': promptMsg}

        if mImg is not None:
            if mImg.dtype != np.uint8:
                raise ValueError('The image must be a uint8 NumPy array.')

            dMsgUser["content"] = [
                {'type': 'text', 'text': promptMsg},
                {'type': 'image_url', 'image_url': {'url': _EncodeImageToDataUri(mImg)}},
            ]

        oResponse = self.oModelClient.chat.completions.create(
            model = self.modelName,
            messages = [*self.lMessages, dMsgUser],
        )

        oMsg      = oResponse.choices[0].message
        answerMsg = oMsg.content

        if answerMsg is None:
            raise RuntimeError(oMsg.refusal or 'The model did not return a text answer.')

        self.lMessages.extend([dMsgUser, {'role': 'assistant', 'content': answerMsg}])

        return answerMsg

