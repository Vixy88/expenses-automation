from PIL import Image, ImageDraw, ImageFont
import os

def text_to_image(text_file, image_file, width=800, height=1200):
    """Convert text file content to an image file"""
    # Create a blank white image
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font - this will work on most systems
    try:
        font = ImageFont.truetype("Arial", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Read the text file
    with open(text_file, 'r') as f:
        text = f.read()
    
    # Draw the text on the image
    draw.text((20, 20), text, fill='black', font=font)
    
    # Save the image
    image.save(image_file)
    print(f"Created image: {image_file}")

# Convert text files to images
current_dir = os.path.dirname(os.path.abspath(__file__))
text_to_image(os.path.join(current_dir, 'taxi_receipt.txt'), 
              os.path.join(current_dir, 'taxi_expense.jpeg'))
text_to_image(os.path.join(current_dir, 'restaurant_receipt.txt'), 
              os.path.join(current_dir, 'dinner_business_expense.jpeg'))

print("Test files created successfully!") 