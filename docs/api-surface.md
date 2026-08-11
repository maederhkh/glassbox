# API surface

Probed model: `openai/gpt-5-mini`. Regenerate with `python scripts/probe_api.py`.

| attempt | works | notes |
|---|---|---|
| `chat: plain` | yes |  |
| `chat: temperature=0.7` | yes |  |
| `chat: max_completion_tokens=512` | yes |  |
| `chat: reasoning_effort=low` | yes |  |
| `chat: reasoning_effort=high` | yes |  |
| `chat: extra_body reasoning effort=low` | yes |  |
| `chat: extra_body reasoning effort=high` | yes |  |
| `slug resolves: openai/gpt-4o` | yes |  |
| `slug resolves: deepseek/deepseek-chat` | yes |  |
| `slug resolves: qwen/qwen3-32b` | yes |  |

```json
[
  {
    "attempt": "chat: plain",
    "ok": true,
    "usage": {
      "completion_tokens": 179,
      "prompt_tokens": 20,
      "total_tokens": 199,
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
      "cost": 0.000363,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000363,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000358
      }
    }
  },
  {
    "attempt": "chat: temperature=0.7",
    "ok": true,
    "usage": {
      "completion_tokens": 145,
      "prompt_tokens": 20,
      "total_tokens": 165,
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
      "cost": 0.000295,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000295,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.00029
      }
    }
  },
  {
    "attempt": "chat: max_completion_tokens=512",
    "ok": true,
    "usage": {
      "completion_tokens": 167,
      "prompt_tokens": 20,
      "total_tokens": 187,
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
      "cost": 0.000339,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000339,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000334
      }
    }
  },
  {
    "attempt": "chat: reasoning_effort=low",
    "ok": true,
    "usage": {
      "completion_tokens": 170,
      "prompt_tokens": 20,
      "total_tokens": 190,
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
      "cost": 0.000345,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000345,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.00034
      }
    }
  },
  {
    "attempt": "chat: reasoning_effort=high",
    "ok": true,
    "usage": {
      "completion_tokens": 535,
      "prompt_tokens": 20,
      "total_tokens": 555,
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
      "cost": 0.001075,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.001075,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.00107
      }
    }
  },
  {
    "attempt": "chat: extra_body reasoning effort=low",
    "ok": true,
    "usage": {
      "completion_tokens": 149,
      "prompt_tokens": 20,
      "total_tokens": 169,
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
      "cost": 0.000303,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.000303,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.000298
      }
    }
  },
  {
    "attempt": "chat: extra_body reasoning effort=high",
    "ok": true,
    "usage": {
      "completion_tokens": 592,
      "prompt_tokens": 20,
      "total_tokens": 612,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 512,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 0.001189,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 0.001189,
        "upstream_inference_prompt_cost": 5e-06,
        "upstream_inference_completions_cost": 0.001184
      }
    }
  },
  {
    "attempt": "slug resolves: openai/gpt-4o",
    "ok": true,
    "usage": {
      "completion_tokens": 2,
      "prompt_tokens": 14,
      "total_tokens": 16,
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
      "cost": 5.5e-05,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 5.5e-05,
        "upstream_inference_prompt_cost": 3.5e-05,
        "upstream_inference_completions_cost": 2e-05
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
      "cost": 4.98e-06,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 4.98e-06,
        "upstream_inference_prompt_cost": 3.2e-06,
        "upstream_inference_completions_cost": 1.78e-06
      }
    }
  },
  {
    "attempt": "slug resolves: qwen/qwen3-32b",
    "ok": true,
    "usage": {
      "completion_tokens": 92,
      "prompt_tokens": 16,
      "total_tokens": 108,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": 0,
        "reasoning_tokens": 89,
        "rejected_prediction_tokens": null,
        "image_tokens": 0
      },
      "prompt_tokens_details": {
        "audio_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "video_tokens": 0
      },
      "cost": 2.92e-05,
      "is_byok": false,
      "cost_details": {
        "upstream_inference_cost": 2.92e-05,
        "upstream_inference_prompt_cost": 1.6e-06,
        "upstream_inference_completions_cost": 2.76e-05
      }
    }
  }
]
```
