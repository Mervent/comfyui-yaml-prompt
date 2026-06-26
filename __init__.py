try:
    # ComfyUI loads custom_nodes/<folder> as a sub-package → relative import works
    from .yaml_prompt.node import YAMLPromptLoader
    from .yaml_prompt.apply_lora import ApplyLoraStack
except ImportError:
    # Standalone / pytest context → no parent package, use absolute import
    from yaml_prompt.node import YAMLPromptLoader
    from yaml_prompt.apply_lora import ApplyLoraStack

NODE_CLASS_MAPPINGS = {
    "YAMLPromptParser": YAMLPromptLoader,
    "YAMLApplyLoraStack": ApplyLoraStack,
}
