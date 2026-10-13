# Python STD
from enum import auto, Enum, unique
from copy import deepcopy
# import math

from dataclasses import dataclass
import base64
import io
import json
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
from openai import NOT_GIVEN, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionUserMessageParam

# Typing
from typing import Any, Callable, Dict, Final, Generator, Literal, List, Optional, Self, Set, Tuple, Union, get_args
from numpy.typing import NDArray


ReasoningEffort     = Literal['none', 'low', 'medium', 'high', 'xhigh', 'max']
ContextWindowSize   = Literal[400_000, 1_050_000]
ContextWindowBudget = Literal[16_384, 65_536, 262_144]

MODEL_CONTEXT_WINDOW_SIZE: Final[Dict[str, ContextWindowSize]] = {
    'gpt-5.4-nano': 400_000,
    'gpt-5.4-nano-2026-03-17': 400_000,
    'gpt-6-luna': 1_050_000,
    'gpt-6.1-sol': 1_050_000,
}

TIME_REGEX: Final[re.Pattern[str]] = re.compile(r"^\d{2}:\d{2}$")

# Models:
# - OpenAI (https://developers.openai.com/api/docs/models)
#   - gpt-6.1-sol
#   - gpt-6-luna
#   - gpt-5.4-nano

# |                          | GPT 5.4 Nano                   | GPT 6 Luna                          | GPT 6.1 Sol                   |
# |--------------------------|--------------------------------|-------------------------------------|-------------------------------|
# | Identifier               | `gpt-5.4-nano`                 | `gpt-6-luna`                        | `gpt-6.1-sol`                 |
# | Input Modalities         | Text, Image                    | Text, Image                         | Text, Image                   |
# | Output Modalities        | Text                           | Text                                | Text                          |
# | Structured Answer        | Supported                      | Supported                           | Supported                     |
# | Reasoning Effort         | none, low, medium, high, xhigh | none, low, medium, high, xhigh, max | low, medium, high, xhigh, max |
# | Default Reasoning Effort | none                           | medium                              | medium                        |
# | Context Window (Tokens)  | 400,000                        | 1,050,000                           | 1,050,000                     |
# | Temperature (API Range)  | [0, 2]                         | [0, 2]                              | [0, 2]                        |
#
# Remarks:
#  - GPT 5.4 Nano and GPT 6 Luna support temperature only with reasoning effort `none`.
#  - GPT 6.1 Sol does not support temperature.
#  - Reasoning effort `none` disables reasoning.

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
    def __init__( self, apiKey: str, modelName: str, /, *, systemMsg: Optional[str] = None, contextWindowBudget: Optional[ContextWindowBudget] = None, reasoningEffort: Optional[ReasoningEffort] = None, valTemp: Optional[float] = None ) -> None:

        if valTemp is not None and not 0 <= valTemp <= 2:
            raise ValueError('Temperature must be between 0 and 2.')

        if contextWindowBudget is not None and contextWindowBudget not in get_args(ContextWindowBudget):
            raise ValueError(f'Unsupported context window budget: {contextWindowBudget!r}.')

        self._modelName    = modelName
        self._systemMsg    = systemMsg
        self._valTemp      = valTemp
        self._contextWindowSize = MODEL_CONTEXT_WINDOW_SIZE.get(modelName)
        self._contextWindowBudget = contextWindowBudget
        self._contextBudgetExceeded = False
        self.SetReasoningEffort(reasoningEffort)
        self._oModelClient = OpenAI(api_key = apiKey)
        self._lMessages: List[ChatCompletionMessageParam] = [] #<! Session history

        if systemMsg is not None:
            self._lMessages.append({'role': 'system', 'content': systemMsg})

    def Prompt( self ) -> Any:

        raise NotImplementedError('The `Prompt()` method must be implemented by subclasses.')

    def Close( self ) -> None:
        self._oModelClient.close()

    def GetModelName( self ) -> str:
        
        return self._modelName

    def GetSystemMessage( self ) -> Optional[str]:
        
        return self._systemMsg

    def GetSessionHistory( self ) -> List[ChatCompletionMessageParam]:
        
        return self._lMessages.copy()

    def GetTemperature( self ) -> Optional[float]:
        
        return self._valTemp

    def SetTemperature( self, valTemp: float ) -> None:

        if not 0 <= valTemp <= 2:
            raise ValueError('Temperature must be between 0 and 2.')

        self._valTemp = valTemp

    def GetReasoningEffort( self ) -> Optional[ReasoningEffort]:

        return self._reasoningEffort

    def SetReasoningEffort( self, reasoningEffort: Optional[ReasoningEffort] ) -> None:

        if reasoningEffort is not None and reasoningEffort not in get_args(ReasoningEffort):
            raise ValueError(f'Unsupported reasoning effort: {reasoningEffort!r}.')

        if self._modelName == 'gpt-6.1-sol' and reasoningEffort == 'none':
            raise ValueError('GPT-6.1 Sol does not support none reasoning effort.')

        if self._modelName in ('gpt-5.4-nano', 'gpt-5.4-nano-2026-03-17') and reasoningEffort == 'max':
            raise ValueError('GPT-5.4 nano does not support max reasoning effort.')

        self._reasoningEffort = reasoningEffort

    def GetContextWindowSize( self ) -> Optional[ContextWindowSize]:

        return self._contextWindowSize

    def GetContextWindowBudget( self ) -> Optional[ContextWindowBudget]:

        return self._contextWindowBudget

    def _CheckContextBudget( self, oResponse: Optional[ChatCompletion] = None ) -> None:

        if self._contextBudgetExceeded:
            raise RuntimeError('The context window budget was exceeded. Open a new session.')

        if self._contextWindowBudget is None or oResponse is None:
            return

        if oResponse.usage is None:
            raise RuntimeError('The API did not return token usage. Cannot check the context window budget.')

        numTokens = oResponse.usage.total_tokens
        if numTokens > self._contextWindowBudget:
            self._contextBudgetExceeded = True
            raise RuntimeError(f'Context window budget exceeded: {numTokens} > {self._contextWindowBudget} tokens. Open a new session.')



class LLMSessionProse(LLMSession):
    def __init__( self, apiKey: str, modelName: str, /, *, systemMsg: Optional[str] = None, contextWindowBudget: Optional[ContextWindowBudget] = None, reasoningEffort: Optional[ReasoningEffort] = None, valTemp: Optional[float] = None ) -> None:

        super().__init__(apiKey, modelName, systemMsg = systemMsg, contextWindowBudget = contextWindowBudget, reasoningEffort = reasoningEffort, valTemp = valTemp)

    def Prompt( self, promptMsg: str, mImg: Optional[NDArray[np.uint8]] = None ) -> str:

        self._CheckContextBudget()
        dMsgUser: ChatCompletionUserMessageParam = {'role': 'user', 'content': promptMsg}

        if mImg is not None:
            if mImg.dtype != np.uint8:
                raise ValueError('The image must be a `UINT8` NumPy array.')

            dMsgUser["content"] = [
                {'type': 'text', 'text': promptMsg},
                {'type': 'image_url', 'image_url': {'url': _EncodeImageToDataUri(mImg)}},
            ]

        oResponse = self._oModelClient.chat.completions.create(
            model = self._modelName,
            messages = [*self._lMessages, dMsgUser],
            temperature = self._valTemp if self._valTemp is not None else NOT_GIVEN,
            reasoning_effort = self._reasoningEffort if self._reasoningEffort is not None else NOT_GIVEN,
        )

        self._CheckContextBudget(oResponse)
        oMsg      = oResponse.choices[0].message
        answerMsg = oMsg.content

        if answerMsg is None:
            raise RuntimeError(oMsg.refusal or 'The model did not return a text answer.')

        self._lMessages.extend([dMsgUser, {'role': 'assistant', 'content': answerMsg}])

        return answerMsg


class LLMSessionStructured(LLMSession):
    def __init__( self, apiKey: str, modelName: str, /, *, dSchema: Dict[str, Any], systemMsg: Optional[str] = None, contextWindowBudget: Optional[ContextWindowBudget] = None, reasoningEffort: Optional[ReasoningEffort] = None, valTemp: Optional[float] = None ) -> None:

        super().__init__(apiKey, modelName, systemMsg = systemMsg, contextWindowBudget = contextWindowBudget, reasoningEffort = reasoningEffort, valTemp = valTemp)
        self.dSchema = deepcopy(dSchema)

    def UpdateSchema( self, dSchema: Dict[str, Any] ) -> None:

        if dSchema.get('type') != 'object':
            raise ValueError('The response schema must define a JSON object.')

        self.dSchema = deepcopy(dSchema)

    def Prompt( self, promptMsg: str, mImg: Optional[NDArray[np.uint8]] = None ) -> Dict[str, Any]:

        self._CheckContextBudget()
        dMsgUser: ChatCompletionUserMessageParam = {'role': 'user', 'content': promptMsg}

        if mImg is not None:
            if mImg.dtype != np.uint8:
                raise ValueError('The image must be a `UINT8` NumPy array.')

            dMsgUser["content"] = [
                {'type': 'text', 'text': promptMsg},
                {'type': 'image_url', 'image_url': {'url': _EncodeImageToDataUri(mImg)}},
            ]

        oResponse = self._oModelClient.chat.completions.create(
            model = self._modelName,
            messages = [*self._lMessages, dMsgUser],
            temperature = self._valTemp if self._valTemp is not None else NOT_GIVEN,
            reasoning_effort = self._reasoningEffort if self._reasoningEffort is not None else NOT_GIVEN,
            response_format = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'structured_response',
                    'schema': self.dSchema,
                    'strict': True,
                },
            },
        )

        oChoice   = oResponse.choices[0]
        self._CheckContextBudget(oResponse)
        oMsg      = oChoice.message
        answerMsg = oMsg.content

        if oMsg.refusal is not None:
            raise RuntimeError(oMsg.refusal)

        if oChoice.finish_reason != 'stop':
            raise RuntimeError(f'The model did not complete the structured answer: {oChoice.finish_reason!r}.')

        if answerMsg is None:
            raise RuntimeError('The model did not return a structured answer.')

        dAnswer = json.loads(answerMsg)
        if not isinstance(dAnswer, dict):
            raise ValueError('The model did not return a JSON object.')

        self._lMessages.extend([dMsgUser, {'role': 'assistant', 'content': answerMsg}])

        return dAnswer

