import os
import re
import logging
import base64
import requests
import time
import random # Added for randomness!
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from io import BytesIO

# Telegram imports
from telegram import Update
from telegram import InputFile
from telegram import InlineKeyboardMarkup
from telegram import InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import filters
from telegram.ext import ContextTypes
from telegram.ext import CallbackQueryHandler
from telegram.ext import PicklePersistence
from telegram.error import BadRequest

# OpenAI import
from openai import OpenAI

# --- 1. Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FORWARDING_GROUP_ID = os.getenv("FORWARDING_GROUP_ID")
PROCESSOR_API_URL = os.getenv("PROCESSOR_API_URL")
TELEGRAM_COMMAND = "fry"

# --- 2. Initialize APIs ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- 3. Helper Functions ---
def prepare_image_for_editing(image_path, size=1024):
    """Resizes and crops an image to a square, converting to RGBA and saving in place."""
    with Image.open(image_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        background = Image.new('RGBA', img.size, (255, 255, 255))
        background.paste(img, (0, 0), img)
        img = background.convert('RGB')

        w, h = img.size
        short_side = min(w, h)
        left = (w - short_side) / 2
        top = (h - short_side) / 2
        right = (w + short_side) / 2
        bottom = (h + short_side) / 2
        img = img.crop((left, top, right, bottom))

        if img.size[0] != size:
            img = img.resize((size, size), Image.Resampling.LANCZOS)

        img.save(image_path, "PNG")


async def download_telegram_image(file_id, context: ContextTypes.DEFAULT_TYPE, filename="temp_telegram_image.png"):
    """Downloads an image from Telegram and saves it to a file."""
    try:
        telegram_file = await context.bot.get_file(file_id)
        await telegram_file.download_to_drive(filename)
        logging.info(f"Downloaded Telegram image {file_id} to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error downloading Telegram image {file_id}: {e}")
        return None

def generate_image_openai(prompt_prefix, size="1024x1024", quality="standard"):
    """Generates an image using OpenAI."""
    full_prompt = prompt_prefix
    try:
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            size=size,
            quality=quality,
            n=1,
            response_format="b64_json"
        )
        image_base64 = response.data[0].b64_json
        logging.info(f"Generated image for prompt: '{full_prompt[:50]}...'")
        return base64.b64decode(image_base64)
    except Exception as e:
        logging.error(f"Error generating image with OpenAI: {e}")
        return None

def edit_image_openai(image_path, prompt):
    """Edits an image using OpenAI."""
    try:
        with open(image_path, "rb") as img_file:
            response = openai_client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
        image_base64 = response.data[0].b64_json
        logging.info(f"Edited image for prompt: '{prompt[:50]}...'")
        return base64.b64decode(image_base64)
    except Exception as e:
        logging.error(f"Error editing image with OpenAI: {e}")
        return None

def combine_images_side_by_side(image1_path: str, image2_path: str) -> BytesIO:
    """
    Combines two images side-by-side on a 2:1 canvas.
    Assumes input images are already processed to be square (e.g., 1024x1024).
    """
    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")

        # Ensure both images are 1024x1024 for consistent combining
        if img1.size[0] != 1024 or img1.size[1] != 1024:
            img1 = img1.resize((1024, 1024), Image.Resampling.LANCZOS)
        if img2.size[0] != 1024 or img2.size[1] != 1024:
            img2 = img2.resize((1024, 1024), Image.Resampling.LANCZOS)

        combined_width = img1.width + img2.width
        combined_height = max(img1.height, img2.height) # Should be 1024 if both are 1024

        combined_image = Image.new('RGB', (combined_width, combined_height))
        combined_image.paste(img1, (0, 0))
        combined_image.paste(img2, (img1.width, 0))

        output_stream = BytesIO()
        combined_image.save(output_stream, format="PNG")
        output_stream.seek(0) # Rewind the stream to the beginning
        return output_stream
    except Exception as e:
        logging.error(f"Error combining images: {e}", exc_info=True)
        return None

# NEW FUNCTION FOR DEEP FRYING with randomized levels
def apply_deep_fry_effect(image_bytes: bytes) -> BytesIO:
    """
    Applies a 'deep-fried' meme effect to an image with randomized intensity,
    ensuring a minimum intensity level for each effect.
    :param image_bytes: The image data as bytes (e.g., from OpenAI API).
    :return: A BytesIO object containing the deep-fried image as PNG.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # --- Randomize Effect Intensities with a minimum "30" level ---
        # Brightness: Minimum 1.3 (30% brighter than original) to 3.0 (very bright)
        brightness_factor = random.uniform(1.3, 3.0)

        # Contrast: Minimum 1.3 (30% more contrast) to 4.5 (very high contrast)
        contrast_factor = random.uniform(1.3, 4.5)

        # Saturation: Minimum 1.3 (30% more saturation) to 4.0 (very high saturation)
        saturation_factor = random.uniform(1.3, 4.0)

        # Sharpen Repetitions: Minimum 1 to 4 times (ensures some sharpening)
        sharpen_repetitions = random.randint(1, 4)

        # JPEG Quality: Lower number means HIGHER intensity of fry effect (more artifacts)
        # So, 'lowest intensity 30' means the HIGHEST quality number allowed is 30,
        # meaning less intense compression. We'll set the range from 10 (very fried) to 40 (less fried).
        jpeg_quality = random.randint(10, 40)

        # Compression Repetitions: Minimum 3 to 15 times (ensures cumulative artifacting)
        compression_repetitions = random.randint(3, 15)

        logging.info(f"Applying deep-fry with randomized settings: "
                     f"Brightness={brightness_factor:.2f}, "
                     f"Contrast={contrast_factor:.2f}, "
                     f"Saturation={saturation_factor:.2f}, "
                     f"Sharpen Repeats={sharpen_repetitions}, "
                     f"JPEG Quality={jpeg_quality}, "
                     f"Compression Repeats={compression_repetitions}")

        # --- Apply the "Deep Fried" effects with randomized values ---
        # 1. Increase Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

        # Corrected lines:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor) # <--- Corrected!

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor) # <--- Corrected!

        # 4. Sharpen
        for _ in range(sharpen_repetitions):
            img = img.filter(ImageFilter.SHARPEN)

        # 5. JPEG Compression (Repeated) - Most crucial for "deep-fried"
        for _ in range(compression_repetitions):
            temp_stream = BytesIO()
            img.save(temp_stream, format="JPEG", quality=jpeg_quality)
            temp_stream.seek(0)
            img = Image.open(temp_stream).convert("RGB")

        output_stream = BytesIO()
        img.save(output_stream, format="PNG")
        output_stream.seek(0)
        logging.info("Successfully applied randomized deep-fry effect.")
        return output_stream

    except Exception as e:
        logging.error(f"Error applying randomized deep-fry effect: {e}", exc_info=True)
        return None

# NEW FUNCTION FOR ADDING MEME TEXT (Correctly placed and indented)
def add_meme_text(image_stream: BytesIO, top_text: str = "", bottom_text: str = "") -> BytesIO:
    """
    Adds Impact font text with black fill and a white stroke to the top and bottom
    of the image, with text size scaled to 8% of the image height.
    :param image_stream: BytesIO object containing the image.
    :param top_text: Text to display at the top.
    :param bottom_text: Text to display at the bottom.
    :return: BytesIO object with the image and text.
    """
    img = Image.open(image_stream).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Calculate font size based on 8% of image height
    # We multiply by a factor (e.g., 1.25) because font height in Pillow isn't a direct 1:1 with size value
    # and we want the *visual* height of the text to be around 8%.
    # This might need slight tweaking based on font metrics.
    target_font_height_pixels = int(height * 0.08)
    font_size = int(target_font_height_pixels * 1.25) # Adjust this multiplier if needed for visual fit

    # Define font properties
    # Try to load Impact font. If not found, use a default sans-serif font
    try:
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "impact.ttf"), font_size)
    except IOError:
        logging.warning("impact.ttf not found, falling back to default font.")
        # Fallback for local testing or systems without Impact
        # Using a default system font that's usually available
        if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        elif os.path.exists("arial.ttf"): # For Windows environments
            font = ImageFont.truetype("arial.ttf", font_size)
        else:
            font = ImageFont.load_default() # Absolute last resort
            # When using load_default(), the font_size parameter is ignored.
            # You might get a very small font if this fallback is hit.
            # For robust production, ensure Impact.ttf or a good alternative is available.
            logging.error("No suitable truetype font found, using default which might be too small.")


    text_color = (0, 0, 0)  # Black fill
    stroke_color = (255, 255, 255) # White stroke
    stroke_width = max(1, int(font_size * 0.04)) # Scale stroke width with font size, min 1px

    # Function to draw text with stroke
    def draw_text_with_stroke(draw_obj, text, position, font, text_fill, stroke_fill, stroke_width):
        x, y = position
        # Draw stroke
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                # Only draw stroke pixels within a circular radius for smoother stroke
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    draw_obj.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
        # Draw main text
        draw_obj.text(position, text, font=font, fill=text_fill)


    # Calculate text position for top text
    if top_text:
        # Get actual text bounding box size for accurate centering and vertical positioning
        bbox = draw.textbbox((0,0), top_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) / 2
        y = int(height * 0.01) # Small padding from top, also scaled
        draw_text_with_stroke(draw, top_text, (x, y), font, text_color, stroke_color, stroke_width)

    # Calculate text position for bottom text
    if bottom_text:
        # Get actual text bounding box size
        bbox = draw.textbbox((0,0), bottom_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) / 2
        y = height - text_height - int(height * 0.03) # Small padding from bottom, also scaled
        draw_text_with_stroke(draw, bottom_text, (x, y), font, text_color, stroke_color, stroke_width)

    output_stream = BytesIO()
    img.save(output_stream, format="PNG")
    output_stream.seek(0)
    return output_stream


# --- Prompt Definitions ---
AI_DISCLAIMER_PROMPT = "This is an AI-generated image. Please apply the following filter, maintaining the core subject but altering the style as described."

LESS_EXTREME_PROMPT_CORE = (
    "Apply a surreal, exaggerated cosmetic surgery filter to the face in this image. "
    "Make the lips very large and glossy, puff out the cheeks unnaturally, "
    "give the chin an artificial, implant-like shape, and smooth out the skin with a vibrant, almost artificial, a strong orange, highly reflective, and lacquered tone, mimicking an extreme, glossy tan. "
    "Add an overall hyper-waxy, intensely shiny, and taut texture, with skin unnaturally stretched and glistening with a liquid-like sheen, mimicking extreme plastic surgery. "
    "Maintain original facial features for recognition, but with heavily distorted 'bogged' style appearance."
)
MIDDLE_GROUND_PROMPT_CORE = (
    "Apply a noticeable, exaggerated cosmetic surgery filter to the face in this image. "
    "Make the lips significantly large and glossy, puff out the cheeks to a distinct, unnatural degree, "
    "give the chin a pronounced, artificial shape, and smooth out the skin with a slightly discolored, yellowish-orange tone, with some emphasis on texture. "
    "Add an overall waxy and somewhat taut texture, with skin unnaturally stretched and shiny, mimicking noticeable plastic surgery. "
    "Maintain original facial features for recognition, but with clearly distorted 'bogged' style appearance."
)
MORE_EXTREME_PROMPT_CORE = (
    "Apply a surreal, exaggerated cosmetic surgery filter to the face in this image"
    "Make the lips very large and glossy, puff out the cheeks unnaturally, "
    "give the chin an artificial, implant-like shape, and smooth out the skin with a discolored tan tone, with the orange hue subtly reduced. "
    "Add an overall waxy texture, with skin unnaturally stretched and shiny, mimicking extreme plastic surgery. "
    "Maintain original facial features for recognition, but with heavily distorted 'bogged' style appearance."
)
OG_PROMPT_CORE = (
    "Apply a surreal, exaggerated cosmetic surgery filter to the face in this image. "
    "Make the lips very large and glossy, puff out the cheeks unnaturally, "
    "give the chin an artificial, implant-like shape, and smooth out the skin with a discolored orange/tan tone. "
    "Add an overall waxy texture, with skin unnaturally stretched and shiny, mimicking extreme plastic surgery. "
    "Maintain original facial features for recognition, but with heavily distorted 'bogged' style appearance."
)
BOGCHAD_PROMPT_CORE = (
    "Apply a surreal, exaggerated cosmetic surgery filter to the face in this image. "
    "Make the lips very large and glossy, puff out the cheeks unnaturally, "
    "give the chin a chiseled jawline, pronounced cheekbones, and a square chin, "
    "and smooth out the skin with a discolored orange/tan tone. "
    "Add an overall waxy texture, with skin unnaturally stretched and shiny, mimicking extreme plastic surgery. "
    "Maintain original facial features for recognition, but with heavily distorted 'bogged' style appearance."
)
GIGABOG_PROMPT_CORE = (
    "Apply a surreal, exaggerated cosmetic surgery filter to the face in this image, rendered entirely in greyscale. "
    "Emphasize an extremely rigid and angular bone structure with stone-like rigidity. "
    "Make the lips very large and glossy, puff out the cheeks unnaturally over sharply defined cheekbones, "
    "and give the jawline and chin a chiseled, square-cut appearance. "
    "Smooth out the skin tones to create a stark contrast with the hard facial features. "
    "Add an overall waxy texture, with skin unnaturally stretched and shiny, mimicking extreme plastic surgery. "
    "Maintain original facial features for recognition, but with a heavily distorted 'bogged' style appearance in black and white."
)
LOG_PROMPT_CORE = (
    "Transform the face in this image to look as if it is carved from a solid log of wood, "
    "but with surreal, exaggerated cosmetic surgery features. Apply a deep, rich wood grain "
    "texture across the entire surface, with visible rings and knots. The facial structure "
    "should be blocky and rigid, yet feature unnaturally large, puffed-out cheeks and very "
    "large, glossy lips that look overfilled. The entire face, including the exaggerated "
    "features, should appear sculpted from a single piece of wood with a skin tone ranging "
    "from light tan to deep oak. Add a semi-gloss varnish or sealant effect, giving the "
    "unnaturally stretched and shiny surface a slight sheen. Maintain basic facial features "
    "for recognition, but presented in a heavily distorted, wooden 'bogged' style."
)
BERRY_PROMPT_CORE = (
    "Apply an intensely surreal, exaggerated cosmetic surgery filter to the face in this image. "
    "Make the lips extremely large, plump, and a vibrant, glossy berry-red, with a wet, reflective sheen. "
    "Puff out the cheeks unnaturally, give the chin an artificial, bulbous shape, "
    "and smooth out the skin with an exaggerated, deep red-orange hue, as if stained by berry juice, with an intense, almost metallic shine. "
    "Add an overall very thick, highly reflective, and sticky-looking waxy texture, "
    "with skin unnaturally stretched, taut, and glistening intensely, mimicking extreme, over-the-top plastic surgery. "
    "Maintain original facial features for recognition, but with a drastically distorted, hyper-glossy, and vividly colored 'bogged' style appearance."
)

def get_prompt_by_key(key):
    prompts = {
        'mini': MORE_EXTREME_PROMPT_CORE,
        'mid': MIDDLE_GROUND_PROMPT_CORE,
        'max': LESS_EXTREME_PROMPT_CORE,
        'og': OG_PROMPT_CORE,
        'bogchad': BOGCHAD_PROMPT_CORE,
        'gigabog': GIGABOG_PROMPT_CORE,
        'log': LOG_PROMPT_CORE,
        'berry': BERRY_PROMPT_CORE
    }
    return prompts.get(key, LESS_EXTREME_PROMPT_CORE)

# --- 4. Telegram Bot Command Handlers ---
async def forward_to_channel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_stream: BytesIO,
    filename: str,
    prompt_key: str,
    is_document: bool = False
):
    """Forwards the generated image to a specified group."""
    if not FORWARDING_GROUP_ID:
        return

    try:
        user = update.effective_user
        caption = (
            f"New generation by {user.mention_html()} "
            f"using prompt: <b>{prompt_key.upper()}</b>"
        )

        image_stream.seek(0)

        if is_document:
            await context.bot.send_document(
                chat_id=FORWARDING_GROUP_ID,
                document=InputFile(image_stream, filename=filename),
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_photo(
                chat_id=FORWARDING_GROUP_ID,
                photo=InputFile(image_stream, filename=filename),
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        logging.info(f"[{user.id}] Successfully forwarded generation to group {FORWARDING_GROUP_ID}.")

    except Exception as e:
        logging.error(f"[{update.effective_user.id}] Failed to forward generation to group {FORWARDING_GROUP_ID}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm the Bogged Bot. "
        f"To get started, send me a picture with the caption `/{TELEGRAM_COMMAND}`. "
        "I'll give you options to apply a surreal cosmetic surgery filter to it!"
    )

async def command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the dynamic command for applying the 'bog' aesthetic."""
    current_time = time.time()
    user_id = update.effective_user.id

    last_command_time = context.user_data.get('last_command_time', 0)

    if current_time - last_command_time < 10:
        logging.warning(f"User {user_id} is rate-limited. Deleting message.")
        try:
            await update.message.delete()
        except BadRequest as e:
            logging.error(f"Failed to delete rate-limited message for user {user_id}: {e}")
        return

    context.user_data['last_command_time'] = current_time

    if not update.message.photo:
        await update.message.reply_text(f"Please attach an image when using the `/{TELEGRAM_COMMAND}` command.")
        return

    original_photo_file_id = update.message.photo[-1].file_id
    context.user_data['last_bogged_image_id'] = original_photo_file_id
    logging.info(f"[{update.effective_user.id}] Stored original image ID: {original_photo_file_id}")

    temp_path = f"bogged_original_input_{update.effective_message.message_id}.png"
    if not await download_telegram_image(original_photo_file_id, context, temp_path):
        await update.message.reply_text("I couldn't download your image. Please try again.")
        return

    processed_image_stream = BytesIO()
    try:
        with Image.open(temp_path) as img:
            img.save(processed_image_stream, format="PNG")
        processed_image_stream.seek(0)
    except Exception as e:
        logging.error(f"[{update.effective_user.id}] Error preparing image display: {e}", exc_info=True)
        await update.message.reply_text("I had trouble preparing your image for display. Please try again.")
        return
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # THIS IS THE FIRST FIX
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("MINI", callback_data='mini'), InlineKeyboardButton("MID", callback_data='mid'), InlineKeyboardButton("MAX", callback_data='max')],
        [InlineKeyboardButton("OG", callback_data='og'), InlineKeyboardButton("BOGCHAD", callback_data='bogchad'), InlineKeyboardButton("GIGABOG", callback_data='gigabog')],
        [InlineKeyboardButton("LOG", callback_data='log'), InlineKeyboardButton("BERRY", callback_data='berry')]
    ])

    await update.message.reply_photo(
        photo=InputFile(processed_image_stream, filename=f"original_image.png"),
        caption="bog this shit",
        reply_markup=keyboard,
        reply_to_message_id=update.message.message_id
    )

async def process_bogged_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_key: str):
    """Helper function to process and send a bogged image based on prompt_key."""
    query = update.callback_query
    await query.answer(f"Applying {prompt_key.upper()} filter...")

    original_photo_file_id = context.user_data.get('last_bogged_image_id')
    if not original_photo_file_id:
        await query.message.reply_text(f"I couldn't find the original image. Please send a new one with `/{TELEGRAM_COMMAND}`.")
        return

    bog_prompt_core = get_prompt_by_key(prompt_key)
    bog_prompt = AI_DISCLAIMER_PROMPT + bog_prompt_core
    await query.message.reply_text(f"doing the '{prompt_key.upper()}' filter nigga", reply_to_message_id=query.message.message_id)

    temp_path = f"bogged_processing_{query.id}.png"
    if not await download_telegram_image(original_photo_file_id, context, temp_path):
        await query.message.reply_text("I couldn't re-download the original image. Please try again.")
        return

    try:
        prepare_image_for_editing(temp_path)
    except Exception as e:
        logging.error(f"Failed to prepare image {temp_path} for editing: {e}")
        await query.message.reply_text("I had a problem preparing your image for the filter. It might be in an unusual format.")
        os.remove(temp_path)
        return

    image_bytes = edit_image_openai(temp_path, bog_prompt)
    os.remove(temp_path)

    if not image_bytes:
        await query.message.reply_text("image too pozzed try another")
        return

    # --- APPLY DEEP FRY EFFECT HERE ---
    logging.info(f"Applying deep fry effect for prompt key: {prompt_key}")
    deep_fried_image_stream = apply_deep_fry_effect(image_bytes)
    if not deep_fried_image_stream:
        await query.message.reply_text("Failed to apply deep-fry effect. The image might be too complex or already corrupted.")
        # If deep fry fails, revert to the original OpenAI output
        processed_image_stream = BytesIO(image_bytes)
    else:
        processed_image_stream = deep_fried_image_stream
    # -----------------------------------

    # --- ADD MEME TEXT HERE (ALWAYS RANDOM) ---
    # Define a list of possible (top_text, bottom_text) pairs that make sense together
    possible_text_pairs = [
        ("WHEN YOU", "GET BOGGED"),
        ("ME WHEN", "THE BOG HITS"),
        ("MY BRAIN", "AFTER BOGGING"),
        ("IT'S OVER", "WERE BOGGED"),
        ("CERTIFIED", "BOGLIN"),
        ("FEELING", "THE BOG TODAY"),
        ("REAL", "BOG HOURS"),
        ("MY", "BOGIGGA"),
        ("SHE", "OR BOG TRYIN"),
        ("THE BOG", "ISRAEL"),
        ("SHE BOG", "ON MY LIN"),
        ("LOOKS", "MINIMIZING"),
        ("UNBOGLIN", "YOURSELF"),
        ("BIG HOG", "MAX BOG"),
        ("BOG", "KNOWER"),
        ("GET", "BOGGED"),
        ("MAX", "BOGGING")
    ]

    # Always pick a random coherent pair, no prompt-specific overrides
    chosen_pair = random.choice(possible_text_pairs)
    top_text = chosen_pair[0]
    bottom_text = chosen_pair[1]

    logging.info(f"Adding meme text: Top='{top_text}', Bottom='{bottom_text}'")
    final_image_stream = add_meme_text(processed_image_stream, top_text, bottom_text)
    if not final_image_stream:
        await query.message.reply_text("Failed to add meme text. Sending without text.")
        final_image_stream = processed_image_stream # Fallback to image without text
    # --------------------------

    output_filename = f"bogged_output_{query.id}_{prompt_key}_fried_meme.png" # Changed filename to indicate 'fried' and 'meme'

    # THIS IS THE SECOND FIX
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("MINI", callback_data='mini'), InlineKeyboardButton("MID", callback_data='mid'), InlineKeyboardButton("MAX", callback_data='max')],
        [InlineKeyboardButton("OG", callback_data='og'), InlineKeyboardButton("BOGCHAD", callback_data='bogchad'), InlineKeyboardButton("GIGABOG", callback_data='gigabog')],
        [InlineKeyboardButton("LOG", callback_data='log'), InlineKeyboardButton("BERRY", callback_data='berry')],
        [InlineKeyboardButton("🚀 Share on X", callback_data='share_x')]
    ])

    try:
        await query.message.reply_photo(
            photo=InputFile(final_image_stream, filename=output_filename),
            caption=f"'{prompt_key.upper()}' ass nigga (deep-fried meme)", # Added "deep-fried meme" to caption
            reply_markup=keyboard,
            reply_to_message_id=query.message.reply_to_message.message_id
        )
        await forward_to_channel(update, context, final_image_stream, output_filename, prompt_key, is_document=False)

    except BadRequest as e:
        if "Image_process_failed" in str(e):
            logging.warning(f"Photo upload failed. Attempting to send as DOCUMENT.")
            try:
                final_image_stream.seek(0)
                await query.message.reply_document(
                    document=InputFile(final_image_stream, filename=output_filename),
                    caption=f"Here's the '{prompt_key.upper()}' version (deep-fried meme, sent as a document).", # Added "deep-fried meme" to caption
                    reply_markup=keyboard,
                    reply_to_message_id=query.message.reply_to_message.message_id
                )
                await forward_to_channel(update, context, final_image_stream, output_filename, prompt_key, is_document=True)
            except Exception as doc_e:
                logging.error(f"Failed to send image as DOCUMENT: {doc_e}", exc_info=True)
                await query.message.reply_text("I couldn't even send the result as a file. It's truly pozzed.")
        else:
            logging.error(f"Telegram BadRequest error: {e}", exc_info=True)
            await query.message.reply_text("A Telegram error occurred while sending the image.")
    except Exception as e:
        logging.error(f"Unexpected error when sending photo: {e}", exc_info=True)
        await query.message.reply_text("An unexpected error occurred while sending the result.")


async def process_share_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the 'Share on X' button, gets the URL back from the API,
    and replies with a new message containing a URL button.
    """
    query = update.callback_query
    await query.answer("Sending to processing service...")

    file_id = query.message.photo[-1].file_id
    chat_id = query.message.chat_id

    logging.info(f"Share button clicked. Downloading file_id: {file_id}")

    try:
        # Download the file into memory
        telegram_file = await context.bot.get_file(file_id)
        image_stream = BytesIO()
        await telegram_file.download_to_memory(image_stream)
        image_stream.seek(0)

        # Define the file and data for the POST request
        files = {'image_file': ('image.jpg', image_stream, 'image/jpeg')}
        data = {'chat_id': chat_id}

        # Make the API call and get the response
        response = requests.post(PROCESSOR_API_URL, files=files, data=data, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        share_url = response_data.get("share_url")

        if share_url:
            # Create a keyboard with the final URL button
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Share on X", url=share_url)]
            ])

            # Remove the old keyboard from the image message to clean up the UI
            await query.edit_message_reply_markup(reply_markup=None)

            # Send a new message with the final link button
            await context.bot.send_message(
                chat_id=chat_id,
                text="Your link is ready!",
                reply_markup=keyboard
            )
        else:
            raise ValueError("API response did not contain a share_url.")

    except Exception as e:
        logging.error(f"An error occurred in share callback: {e}", exc_info=True)
        await context.bot.send_message(chat_id, "Error: The processing service failed to return a link.")



async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    MODIFIED: Handles all inline keyboard button presses.
    Acts as a router to direct flow based on callback_data.
    """
    query = update.callback_query

    if not query.message.reply_to_message:
        await query.answer("Cannot verify the owner of this menu.", show_alert=True)
        return

    clicker_id = query.from_user.id
    owner_id = query.message.reply_to_message.from_user.id

    if clicker_id != owner_id:
        await query.answer("This menu isn't for you.", show_alert=True)
        return

    # --- Routing Logic ---
    if query.data == 'share_x':
        await process_share_callback(update, context)
    else:
        await process_bogged_image(update, context, query.data)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and a user-friendly message."""
    logging.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    if update and hasattr(update, 'effective_message') and update.effective_message:
        await update.effective_message.reply_text("An unexpected error occurred. Please try again later.")

# --- 5. Main Execution Block ---
def main() -> None:
    """Start the bot."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        exit(1)
    if not TELEGRAM_BOT_TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN environment variable is not set.")
        exit(1)

    if not PROCESSOR_API_URL:
        logging.warning("PROCESSOR_API_URL environment variable is not set. The 'Share on X' feature will be disabled.")

    if not FORWARDING_GROUP_ID:
        logging.warning("FORWARDING_GROUP_ID environment variable is not set. Output will not be sent to a group.")
    else:
        logging.info(f"Generations will be forwarded to group ID: {FORWARDING_GROUP_ID}")

    logging.info(f"Starting Telegram image bot with command: /{TELEGRAM_COMMAND}")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    command_regex = re.compile(rf'^/{TELEGRAM_COMMAND}\b', re.IGNORECASE)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & filters.CaptionRegex(command_regex), command_handler))
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    application.add_error_handler(error_handler)

    logging.info("Telegram bot started. Press Ctrl-C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()