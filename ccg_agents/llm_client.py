"""
大语言模型客户端：实现 generate(prompt, **kwargs) 接口，供结构/事实/逻辑等智能体的
可解释模块调用。支持 OpenAI 兼容 API（含本地代理、Azure 等）。
"""

from __future__ import annotations

import os
from typing import Any, Optional


class OpenAIClient:
    """
    基于 OpenAI 或兼容接口的 LLM 客户端，满足 generate(prompt, **kwargs) 协议。
    可通过环境变量 OPENAI_API_KEY、OPENAI_API_BASE 配置（不设则使用库默认）。
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_API_BASE")

    def generate(self, prompt: str, max_tokens: int = 200, **kwargs: Any) -> str:
        """
        发送单轮用户提示，返回模型生成文本。kwargs 会传给 openai 的 create 调用。
        """
        try:
            import openai
        except ImportError:
            return _fallback_explanation()

        client_kwargs: dict = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        if client_kwargs:
            client = openai.OpenAI(**client_kwargs)
        else:
            client = openai.OpenAI()

        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **kwargs,
        }
        resp = client.chat.completions.create(**params)
        if not resp.choices:
            return _fallback_explanation()
        return (resp.choices[0].message.content or "").strip()


def _fallback_explanation() -> str:
    """未配置 API 或调用失败时返回的占位说明，便于本地跑通流程。"""
    return "[未配置 LLM 或调用失败，此处为占位说明；请设置 OPENAI_API_KEY 或传入有效客户端后重新运行。]"


class LocalQwenClient:
    """
    使用本地 transformers 加载的 Qwen2.5 模型，满足 generate(prompt, max_tokens=...) 协议，
    供结构/事实/逻辑等智能体使用。首次调用 generate 时懒加载模型。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        - model_name: Hugging Face 模型名，未提供本地路径时按此名称加载（必要时自动下载）。
        - model_path: 本地模型目录；不传则读取环境变量 LOCAL_LLM_PATH。
        - cache_dir: Hugging Face 缓存目录；不传则读取 HF_HUB_CACHE，都没有则用 HF 默认缓存。
        """
        self.model_name = model_name
        self.model_path = model_path or os.environ.get("LOCAL_LLM_PATH")
        self.cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
        self._device_map = device_map
        self._tokenizer = None
        self._model = None

    def _load(self, source: str, local_files_only: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype="auto",
            device_map=self._device_map,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        # 1) 指定了本地模型目录：直接从该目录加载
        if self.model_path and os.path.exists(self.model_path):
            self._load(self.model_path, local_files_only=True)
            return

        # 2) 否则按模型名加载：先只用本地缓存，缓存不完整时再允许下载
        try:
            self._load(self.model_name, local_files_only=True)
        except Exception:
            self._load(self.model_name, local_files_only=False)

    def generate(self, prompt: str, max_tokens: int = 200, **kwargs: Any) -> str:
        try:
            self._ensure_model()
        except Exception:
            return _fallback_explanation()

        tokenizer = self._tokenizer
        model = self._model
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            **kwargs,
        )
        gen = out[0, inputs["input_ids"].shape[1] :]
        ans = tokenizer.decode(gen, skip_special_tokens=True)
        return ans.strip()


class MockLLMClient:
    """
    不发起真实调用的占位客户端，用于测试或未配置 API 时跑通流程。
    返回固定短句，便于检查解释器与保存逻辑是否正常。
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "（当前为 Mock 客户端：未调用真实大模型，仅作流程测试。）"
