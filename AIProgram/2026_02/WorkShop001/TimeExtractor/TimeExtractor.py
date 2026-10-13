# Python STD
from enum import auto, Enum, unique
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
import skimage as ski

# Optimization

# Python STD

# Auxiliary

# Visualization

# Miscellaneous
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

# Typing
from typing import Any, Callable, Dict, Final, Generator, Literal, List, Optional, Self, Set, Tuple, Union
from numpy.typing import NDArray


TIME_REGEX: Final[re.Pattern[str]] = re.compile(r"^\d{2}:\d{2}$")

# Models:
# - OpenAI (https://developers.openai.com/api/docs/models)
#   - gpt-5.4
#   - gpt-5.4-pro
#   - gpt-5.4-mini
#   - gpt-5.4-nano
# - Gemini (https://ai.google.dev/gemini-api/docs/models)
#   - gemini-3.1-pro-preview
#   - gemini-3-flash-preview
#   - gemini-3.1-flash-lite-preview

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

@dataclass(slots = True)
class TimeExtractionResult:
    hasTime: bool
    matchTime: str | None
    valConfidence: float
    stoppageTimeReferee: str | None
    stoppageTimeActual: str | None
    firstTeamScore: int | None
    secondTeamScore: int | None


class TimeExtractor():
    def __init__( self, apiKey: str, modelName: Literal['gpt-6-luna', 'gpt-5.4-nano', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite-preview'], /, *, resizeFctr: Optional[float] = None, tuCropSize: Optional[Tuple[int, int, int, int]] ) -> None:

        match modelName:
            # Sets `self.dResponseSchema`, `self.dMsgSystem`, `self.oModelClient`, and `self.modelSupplier` based on the model name and API key
            case 'gpt-5.4' | 'gpt-5.4-pro' | 'gpt-5.4-mini' | 'gpt-5.4-nano':
                self.__initOpenAI__(apiKey, modelName)
            case 'gemini-3.1-pro-preview' | 'gemini-3-flash-preview' | 'gemini-3.1-flash-lite-preview':
                self.__initGemini__(apiKey, modelName)
            case _:
                raise ValueError(f'Unsupported model name: {modelName!r}')

        self.resizeFctr = resizeFctr
        self.tuCropSize = tuCropSize

    def __initOpenAI__( self, openAIKey: str, modelName: Literal['gpt-5.4-mini', 'gpt-5.4-nano'] ) -> None:

        # Defining the structured response schema for the model's output
        dResponseSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": "football_time_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "has_time": {
                            "type": "boolean",
                            "description": "Whether a match clock is visible in the frame.",
                        },
                        "time_mm_ss": {
                            "type": ["string", "null"],
                            "description": "Visible match time in MM:SS format, or null when unavailable.",
                            "pattern": "^\\d{2}:\\d{2}$",
                        },
                        "certainty": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence score for the overall extraction.",
                        },
                        "stoppage_time_referee": {
                            "type": ["string", "null"],
                            "description": "Visible stoppage time decided by the referee in +X format, or null when unavailable.",
                            "pattern": "^\\+\\d+$",
                        },
                        "stoppage_time_actual": {
                            "type": ["string", "null"],
                            "description": "Visible actual stoppage time in MM:SS (e.g. `02:19`) or M:SS (e.g. `2:19`) format, or null when unavailable.",
                            "pattern": "^\\d{2}:\\d{2}$",
                        },
                        "first_team_score": {
                            "type": ["integer", "null"],
                            "description": "Visible score of the first team, or null when unavailable.",
                            "pattern": "^\\d+$",
                            "minimum": 0,
                            "maximum": 99,
                        },
                        "second_team_score": {
                            "type": ["integer", "null"],
                            "description": "Visible score of the second team, or null when unavailable.",
                            "pattern": "^\\d+$",
                            "minimum": 0,
                            "maximum": 99,
                        },

                    },
                    "required": ["has_time", "time_mm_ss", "certainty", "stoppage_time_referee", "stoppage_time_actual", "first_team_score", "second_team_score"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        dMsgSystem = {
            "role": "system",
            "content": (
                "You extract the match clock from a single football broadcast frame. "
                "Return only the structured result. Use MM:SS when a valid clock is visible. "
                "The frame may include overtime or stoppage time in the format MM:SS or M:SS. It should be parsed in the structured result. "
                "The frame may also include a referee indicated stoppage time in the format +X, such as +3. It should be parsed in the structured result. "
                "If no readable time exists, set has_time to false and time_mm_ss to null. "
                "The frame may also include the scores of the two teams. The first team's score is the one on the left or at the top in the broadcast, and the second team's score is the one on the right or at the bottom in the broadcast. If scores are not visible, set them to null."
            ),
        }
        
        self.modelName  = modelName
        
        
        self.dResponseSchema = dResponseSchema
        self.dMsgSystem      = dMsgSystem
        
        self.oModelClient = OpenAI(api_key = openAIKey)
        self.modelSupplier = ModelSupplier.OPENAI

        self.modelName  = modelName

        self.dResponseSchema = dResponseSchema
        self.dMsgSystem      = dMsgSystem

    
    def Predict( self, mImg: NDArray ) -> TimeExtractionResult:

        if self.tuCropSize is not None:
            x, y, w, h = self.tuCropSize
            mImg = mImg[y:(y + h), x:(x + w)]

        if self.resizeFctr is not None:
            mImg = ski.transform.rescale(mImg, scale = self.resizeFctr, channel_axis = -1, anti_aliasing = True)
            mImg = ski.util.img_as_ubyte(mImg)

        match self.modelSupplier:
            case ModelSupplier.OPENAI:
                return self._PredictOpenAI(mImg)
            case ModelSupplier.GEMINI:
                return self._PredictGemini(mImg)
    
    def _PredictOpenAI( self, mImg: NDArray ) -> TimeExtractionResult:
        # Image is Pre Processed (cropping, resizing) before being passed to this function.
        
        imageDataUri = _EncodeImageToDataUri(mImg)

        response = self.oModelClient.chat.completions.create(
            model = self.modelName,
            temperature = 0,
            response_format = self.dResponseSchema,
            messages = [
                self.dMsgSystem,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Inspect this football frame and determine whether the broadcast match time is visible. "
                                "Return the detected time and certainty. "
                                "If stoppage time is visible, return the referee indicated stoppage time and the actual stoppage time if visible. "
                                "Return the scores of the two teams if visible. "
                                "Make sure you work according to the structure of the response."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": imageDataUri,
                            },
                        },
                    ],
                },
            ],
        )

        responseText = response.choices[0].message.content
        dResponse    = json.loads(responseText)

        hasTime   = bool(dResponse["has_time"])
        matchTime = dResponse["time_mm_ss"]
        valConfidence = float(dResponse["certainty"])
        stoppageTimeReferee = dResponse.get("stoppage_time_referee")
        stoppageTimeActual = dResponse.get("stoppage_time_actual")
        firstTeamScore = dResponse.get("first_team_score")
        secondTeamScore = dResponse.get("second_team_score")

        if matchTime is not None and not TIME_REGEX.fullmatch(matchTime):
            raise ValueError(f"The model returned an invalid time format: {matchTime!r}")

        if not hasTime:
            matchTime = None

        valConfidence = min(max(valConfidence, 0.0), 1.0)

        return TimeExtractionResult(
            hasTime = hasTime,
            matchTime = matchTime,
            valConfidence = valConfidence,
            stoppageTimeReferee = stoppageTimeReferee,
            stoppageTimeActual = stoppageTimeActual,
            firstTeamScore = firstTeamScore,
            secondTeamScore = secondTeamScore,
        )



class TimeExtractorOpenAI():
    def __init__( self, openAIKey: str, modelName: Literal['gpt-5.4-mini', 'gpt-5.4-nano'] = 'gpt-5.4-nano', /, *, resizeFctr: Optional[float] = None, tuCropSize: Optional[Tuple[int, int, int, int]] ) -> None:

        # Defining the structured response schema for the model's output
        dResponseSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": "football_time_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "has_time": {
                            "type": "boolean",
                            "description": "Whether a match clock is visible in the frame.",
                        },
                        "time_mm_ss": {
                            "type": ["string", "null"],
                            "description": "Visible match time in MM:SS format, or null when unavailable.",
                            "pattern": "^\\d{2}:\\d{2}$",
                        },
                        "certainty": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence score for the overall extraction.",
                        },
                        "stoppage_time_referee": {
                            "type": ["string", "null"],
                            "description": "Visible stoppage time decided by the referee in +X format, or null when unavailable.",
                            "pattern": "^\\+\\d+$",
                        },
                        "stoppage_time_actual": {
                            "type": ["string", "null"],
                            "description": "Visible actual stoppage time in MM:SS (e.g. `02:19`) or M:SS (e.g. `2:19`) format, or null when unavailable.",
                            "pattern": "^\\d{2}:\\d{2}$",
                        },
                        "first_team_score": {
                            "type": ["integer", "null"],
                            "description": "Visible score of the first team, or null when unavailable.",
                            "pattern": "^\\d+$",
                            "minimum": 0,
                            "maximum": 99,
                        },
                        "second_team_score": {
                            "type": ["integer", "null"],
                            "description": "Visible score of the second team, or null when unavailable.",
                            "pattern": "^\\d+$",
                            "minimum": 0,
                            "maximum": 99,
                        },

                    },
                    "required": ["has_time", "time_mm_ss", "certainty", "stoppage_time_referee", "stoppage_time_actual", "first_team_score", "second_team_score"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        dMsgSystem = {
            "role": "system",
            "content": (
                "You extract the match clock from a single football broadcast frame. "
                "Return only the structured result. Use MM:SS when a valid clock is visible. "
                "The frame may include overtime or stoppage time in the format MM:SS or M:SS. It should be parsed in the structured result. "
                "The frame may also include a referee indicated stoppage time in the format +X, such as +3. It should be parsed in the structured result. "
                "If no readable time exists, set has_time to false and time_mm_ss to null. "
                "The frame may also include the scores of the two teams. The first team's score is the one on the left or at the top in the broadcast, and the second team's score is the one on the right or at the bottom in the broadcast. If scores are not visible, set them to null."
            ),
        }
        
        self.modelName  = modelName
        self.resizeFctr = resizeFctr
        self.tuCropSize = tuCropSize
        
        self.dResponseSchema = dResponseSchema
        self.dMsgSystem      = dMsgSystem
        
        self.openAIClient = OpenAI(api_key = openAIKey)

    def Predict( self, mImg: NDArray ) -> TimeExtractionResult:

        if self.tuCropSize is not None:
            x, y, w, h = self.tuCropSize
            mImg = mImg[y:(y + h), x:(x + w)]

        if self.resizeFctr is not None:
            mImg = ski.transform.rescale(mImg, scale = self.resizeFctr, channel_axis = -1, anti_aliasing = True)
            mImg = ski.util.img_as_ubyte(mImg)
        
        imageDataUri = _EncodeImageToDataUri(mImg)

        response = self.openAIClient.chat.completions.create(
            model = self.modelName,
            temperature = 0,
            response_format = self.dResponseSchema,
            messages = [
                self.dMsgSystem,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Inspect this football frame and determine whether the broadcast match time is visible. "
                                "Return the detected time and certainty. "
                                "If stoppage time is visible, return the referee indicated stoppage time and the actual stoppage time if visible. "
                                "Return the scores of the two teams if visible. "
                                "Make sure you work according to the structure of the response."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": imageDataUri,
                            },
                        },
                    ],
                },
            ],
        )

        responseText = response.choices[0].message.content
        dResponse    = json.loads(responseText)

        hasTime   = bool(dResponse["has_time"])
        matchTime = dResponse["time_mm_ss"]
        valConfidence = float(dResponse["certainty"])
        stoppageTimeReferee = dResponse.get("stoppage_time_referee")
        stoppageTimeActual = dResponse.get("stoppage_time_actual")
        firstTeamScore = dResponse.get("first_team_score")
        secondTeamScore = dResponse.get("second_team_score")

        if matchTime is not None and not TIME_REGEX.fullmatch(matchTime):
            raise ValueError(f"The model returned an invalid time format: {matchTime!r}")

        if not hasTime:
            matchTime = None

        valConfidence = min(max(valConfidence, 0.0), 1.0)

        return TimeExtractionResult(
            hasTime = hasTime,
            matchTime = matchTime,
            valConfidence = valConfidence,
            stoppageTimeReferee = stoppageTimeReferee,
            stoppageTimeActual = stoppageTimeActual,
            firstTeamScore = firstTeamScore,
            secondTeamScore = secondTeamScore,
        )
    
    def ExtractTime( self, timeResult: TimeExtractionResult ) -> Optional[Tuple[int, int]]:
        
        if not timeResult.hasTime or timeResult.matchTime is None:
            
            return None
        
        valMinutes, valSeconds = map(int, timeResult.matchTime.split(':'))
        
        return valMinutes, valSeconds
  