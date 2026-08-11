# API surface

Probed model: `openai/gpt-5-mini`. Regenerate with `python scripts/probe_api.py`.

| attempt | works | notes |
|---|---|---|
| `chat: plain` | yes |  |
| `chat: temperature=0.7` | yes | call did not error -- see 'Temperature support' section below before trusting this |
| `chat: max_completion_tokens=512` | yes |  |
| `chat: reasoning_effort=low` | yes |  |
| `chat: reasoning_effort=high` | yes |  |
| `chat: extra_body reasoning effort=low` | yes |  |
| `chat: extra_body reasoning effort=high` | yes |  |
| `slug resolves: openai/gpt-4o` | yes |  |
| `slug resolves: deepseek/deepseek-chat` | yes |  |
| `slug resolves: qwen/qwen3-32b` | yes |  |

## Temperature support

**Conclusion: NOT HONOURED**, per metadata evidence below. The `chat: temperature=0.7` attempt in the table above returns `ok: true`, but that only shows OpenRouter/the upstream provider *accepted* the request -- not that the parameter had any effect. `temperature` is silently dropped before it reaches the model. Downstream code must not treat 0.7 as a controlled variable: any sampling variation observed in the reliability recipe comes from the model's own default (undocumented) temperature, not from a value this codebase sets. This must be stated as a limitation in the study write-up.

Evidence:

```json
{
  "method": "metadata",
  "temperature_honoured": false,
  "evidence": {
    "model_supported_parameters": [
      "include_reasoning",
      "max_completion_tokens",
      "max_tokens",
      "reasoning",
      "reasoning_effort",
      "response_format",
      "seed",
      "structured_outputs",
      "tool_choice",
      "tools"
    ],
    "model_default_parameters_temperature": null,
    "per_endpoint_supported_parameters": [
      {
        "provider": "OpenAI",
        "supported_parameters": [
          "reasoning",
          "include_reasoning",
          "structured_outputs",
          "response_format",
          "seed",
          "max_tokens",
          "tools",
          "tool_choice",
          "reasoning_effort"
        ]
      },
      {
        "provider": "OpenAI",
        "supported_parameters": [
          "reasoning",
          "include_reasoning",
          "structured_outputs",
          "response_format",
          "seed",
          "max_tokens",
          "tools",
          "tool_choice",
          "reasoning_effort"
        ]
      },
      {
        "provider": "Azure",
        "supported_parameters": [
          "max_completion_tokens",
          "reasoning",
          "include_reasoning",
          "seed",
          "response_format",
          "structured_outputs",
          "tools",
          "tool_choice",
          "reasoning_effort"
        ]
      },
      {
        "provider": "Azure",
        "supported_parameters": [
          "max_completion_tokens",
          "reasoning",
          "include_reasoning",
          "seed",
          "response_format",
          "structured_outputs",
          "tools",
          "tool_choice",
          "reasoning_effort"
        ]
      }
    ]
  }
}
```

## Raw attempt results

```json
[
  {
    "attempt": "chat: plain",
    "ok": true,
    "usage": {
      "completion_tokens": 224,
      "prompt_tokens": 20,
      "total_tokens": 244,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 128,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.000453,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000453,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000448
      }
    }
  },
  {
    "attempt": "chat: temperature=0.7",
    "ok": true,
    "usage": {
      "completion_tokens": 161,
      "prompt_tokens": 20,
      "total_tokens": 181,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 64,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.000327,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000327,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000322
      }
    }
  },
  {
    "attempt": "chat: max_completion_tokens=512",
    "ok": true,
    "usage": {
      "completion_tokens": 205,
      "prompt_tokens": 20,
      "total_tokens": 225,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 128,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.000415,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000415,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.00041
      }
    }
  },
  {
    "attempt": "chat: reasoning_effort=low",
    "ok": true,
    "usage": {
      "completion_tokens": 101,
      "prompt_tokens": 20,
      "total_tokens": 121,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 106,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.000207,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000207,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000202
      }
    }
  },
  {
    "attempt": "chat: reasoning_effort=high",
    "ok": true,
    "usage": {
      "completion_tokens": 536,
      "prompt_tokens": 20,
      "total_tokens": 556,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 448,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.001077,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.001077,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.001072
      }
    }
  },
  {
    "attempt": "chat: extra_body reasoning effort=low",
    "ok": true,
    "usage": {
      "completion_tokens": 118,
      "prompt_tokens": 20,
      "total_tokens": 138,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 109,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.000241,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000241,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000236
      }
    }
  },
  {
    "attempt": "chat: extra_body reasoning effort=high",
    "ok": true,
    "usage": {
      "completion_tokens": 559,
      "prompt_tokens": 20,
      "total_tokens": 579,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 448,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.001123,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.001123,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.001118
      }
    }
  },
  {
    "attempt": "slug resolves: openai/gpt-4o",
    "ok": true,
    "usage": {
      "completion_tokens": 1,
      "prompt_tokens": 14,
      "total_tokens": 15,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 4.5e-05,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 4.5e-05,
        "upstream_inference_prompt_cost": 3.5e-05,
        "upstream_inference_completions_cost": 1e-05
      }
    }
  },
  {
    "attempt": "slug resolves: deepseek/deepseek-chat",
    "ok": true,
    "usage": {
      "completion_tokens": 2,
      "prompt_tokens": 10,
      "total_tokens": 12,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 6.6e-06,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 6.6e-06,
        "upstream_inference_prompt_cost": 4e-06,
        "upstream_inference_completions_cost": 2.6e-06
      }
    }
  },
  {
    "attempt": "slug resolves: qwen/qwen3-32b",
    "ok": true,
    "usage": {
      "completion_tokens": 157,
      "prompt_tokens": 15,
      "total_tokens": 172,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 155,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 9.159e-05,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 9.159e-05,
        "upstream_inference_prompt_cost": 2.1e-06,
        "upstream_inference_completions_cost": 8.949e-05
      }
    }
  }
]
```
