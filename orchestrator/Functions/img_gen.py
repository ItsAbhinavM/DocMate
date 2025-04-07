from dependencies import BaseTool

class imgTool(BaseTool):
    name = "image generation"
    description = "Useful to generating images"

    def __init__(self):
        super().__init__()

    def _run(self, tool_input: str, **kwargs) -> str:
        """Generate image"""
        message = tool_input
        return f"\nquery: {message}\noutput: image generated successfully"

img_gen_tool = imgTool()