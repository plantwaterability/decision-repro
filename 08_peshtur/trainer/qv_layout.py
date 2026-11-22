"""Prompt layout switch of the VL round (arms uoq1 / uoq2). Used by qv_train_vl.py, qv_soup_engine_vl.py and vl_layout_check.py; the stock
qv_common.py / qv_ocr.py are imported unchanged.

images_first  the layout of every arm up to uog1 and of the shipped engine: qv_common.render itself, the OCR block (when a row has one)
              prefixed to the question:   <user> [images] [OCR block] question, options, tail <assistant> (+ readout)
sandwich      <user> question, options (when they are text) [images] [OCR block] question, options, tail <assistant> (+ readout)
              The part in front of the images is the question, a newline, the lettered options one per line, a newline: no OCR block (it
              follows the images it was read from) and no "Answer with ..." tail. From the first image on, the prompt is the images_first
              prompt token for token, so the answer slot (the last prompt token, where the head reads) is unchanged.
              A request without images has nothing to put the question in front of: it is rendered as images_first.
VX (11-18): Gemma-4 port. The layout itself is chat-template content parts (family-agnostic); only the image-start token used to measure the
              header / added tokens was Qwen's <|vision_start|>. For a Gemma processor it is the processor's boi_token (<|image>), and the one
              test image is passed as a nested list (one image list per text). Qwen processors take the old path unchanged.
"""
import qv_common as C

LAYOUTS = ("images_first", "sandwich")
MERGER_KEYS = ("norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias")


def pre_text(question, options, options_in_image=False):
    """What the sandwich layout puts in front of the images."""
    if options_in_image:
        return f"{question}\n"
    opts = "\n".join(f"{C.LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}\n"


def render(proc, question, options, n_images, options_in_image=False, readout=None, ocr_block="", layout="images_first"):
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    if layout == "images_first" or n_images == 0:
        return C.render(proc, ocr_block + question, options, n_images, options_in_image, readout)
    content = [{"type": "text", "text": pre_text(question, options, options_in_image)}]
    content += [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": C.prompt_text(ocr_block + question, options, options_in_image)})
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    return text + readout if readout else text


def header_len(proc):
    """Tokens in front of the first image of an images_first prompt (the chat header), measured on a real encoding."""
    from PIL import Image

    im = Image.new("RGB", (64, 64), (127, 127, 127))
    enc = proc(text=[C.render(proc, "x", ["a", "b"], 1)], images=[[im]] if is_gemma(proc) else [im], return_tensors="pt")
    return first_vision_start(enc["input_ids"][0], proc)


def is_gemma(proc):
    return "Gemma" in type(proc).__name__


def image_start_token(proc):
    """The token that opens an image in the encoded prompt: Gemma-4 the processor's boi_token (<|image>), Qwen <|vision_start|>."""
    if is_gemma(proc):
        t = getattr(proc, "boi_token", None)
        if not t:
            raise RuntimeError("Gemma processor without boi_token")
        return t
    return "<|vision_start|>"


def first_vision_start(ids, proc):
    t = image_start_token(proc)
    vs = proc.tokenizer.convert_tokens_to_ids(t)
    if vs is None or (is_gemma(proc) and vs == proc.tokenizer.unk_token_id):
        raise RuntimeError(f"image-start token {t!r} is not one token")
    hit = (ids == vs).nonzero()
    if hit.numel() == 0:
        raise RuntimeError(f"no {t} in a prompt with images")
    return int(hit[0])


def added_tokens(ids, proc, hlen, layout, n_images):
    """Tokens the sandwich layout adds to a prompt (0 for images_first / no image): the trainer applies its --max-tokens cap to the
    images_first length (n - added), so the layout changes no row's kept / skipped status."""
    if layout != "sandwich" or n_images == 0:
        return 0
    return first_vision_start(ids, proc) - hlen


def merger_fp32(merger):
    """The vision merger with fp32 master weights inside a bf16 model (as the LoRA matrices are fp32): its input is cast to fp32 and its
    output back to the input's dtype. Used by the trainer when the merger is trained and by the eval engine when a checkpoint carries
    merger.safetensors, so both run the same arithmetic."""
    merger.float()

    def pre(mod, args):
        mod._vl_dtype = args[0].dtype
        return (args[0].float(),) + tuple(args[1:])

    def post(mod, args, out):
        return out.to(mod._vl_dtype)

    merger.register_forward_pre_hook(pre)
    merger.register_forward_hook(post)
    return merger


def merger_of(model):
    """model = Qwen3_5ForConditionalGeneration, possibly PEFT-wrapped."""
    mg = C.inner_model(model).visual.merger
    keys = tuple(mg.state_dict().keys())
    if sorted(keys) != sorted(MERGER_KEYS):
        raise RuntimeError(f"unexpected merger parameters: {keys}")
    return mg
