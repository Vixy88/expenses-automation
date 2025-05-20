#!/usr/bin/env python3
import os
import glob
import re
import csv
import pandas as pd
import requests
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import json
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('expense_automation')

# Constants for categories and their keywords
CATEGORIES = {
    'Software & Services': ['github', 'gitlab', 'supabase', 'aws', 'amazon web', 'google cloud', 'azure', 
                           'digital ocean', 'heroku', 'netlify', 'vercel', 'github', 'notion', 'slack', 
                           'atlassian', 'jira', 'confluence', 'trello', 'asana', 'airtable', 'zapier', 
                           'dropbox', 'box', 'zoom', 'microsoft', 'adobe', 'figma', 'canva', 'stripe', 
                           'paypal', 'square', 'shopify', 'mailchimp', 'sendgrid', 'twilio'],
    'Travel': ['airline', 'flight', 'train', 'taxi', 'uber', 'lyft', 'hotel', 'accommodation', 'booking', 'airbnb', 
               'car rental', 'parking', 'toll', 'transport', 'transit', 'travel'],
    'Office Supplies': ['stationery', 'paper', 'printer', 'ink', 'toner', 'pen', 'notebook', 'office', 'desk', 
                        'chair', 'monitor', 'keyboard', 'mouse', 'usb', 'cable', 'adapter', 'charger'],
    'Hardware': ['computer', 'laptop', 'monitor', 'server', 'router', 'switch', 'network', 'hard drive', 'ssd', 
                'memory', 'ram', 'cpu', 'gpu', 'keyboard', 'mouse', 'headphones', 'camera', 'microphone', 'hardware'],
    'Meals & Entertainment': ['restaurant', 'cafe', 'coffee', 'lunch', 'dinner', 'breakfast', 'food', 'meal', 
                             'catering', 'delivery', 'uber eats', 'doordash', 'grubhub', 'entertainment'],
    'Telecommunications': ['phone', 'mobile', 'cell', 'internet', 'broadband', 'fiber', 'wifi', 'data', 'telecom', 
                          'telecommunication', 'voip', 'landline'],
    'Rent & Utilities': ['rent', 'lease', 'office space', 'water', 'electricity', 'gas', 'heating', 'cooling', 
                        'utility', 'utilities'],
    'Professional Services': ['lawyer', 'legal', 'accountant', 'accounting', 'consultant', 'consulting', 'advisor', 
                             'service', 'professional', 'freelance', 'contractor', 'recruitment', 'hr'],
    'Marketing & Advertising': ['ad', 'ads', 'advert', 'advertising', 'marketing', 'promotion', 'campaign', 'seo', 
                               'sem', 'social media', 'branding', 'pr', 'public relations'],
    'Subscriptions': ['subscription', 'monthly', 'annual', 'recurring', 'membership', 'license'],
    'Insurance': ['insurance', 'coverage', 'policy', 'premium', 'health', 'liability', 'property'],
    'Miscellaneous': []  # Default category
}

# Portuguese tax-optimized categories for freelancers (Categories aligned with Simplified Regime deductions)
PORTUGAL_TAX_CATEGORIES = {
    # 15% coefficient category (75% deductible - higher deduction rate)
    'Material Costs & Supplies': ['office supplies', 'stationery', 'paper', 'printer', 'ink', 'toner', 'supplies'],
    'Equipment & Technology': ['computer', 'laptop', 'hardware', 'monitor', 'server', 'router', 'keyboard', 'mouse', 
                              'headphones', 'camera', 'microphone', 'software', 'subscription', 'license', 'github', 
                              'hosting', 'cloud', 'aws', 'azure', 'digital ocean', 'netlify', 'vercel', 'heroku'],
    'Workspace Expenses': ['rent', 'lease', 'office', 'coworking', 'water', 'electricity', 'gas', 'heating', 'cooling',
                          'utilities', 'internet', 'wifi', 'broadband', 'telephone', 'mobile'],
    'Professional Development': ['course', 'training', 'book', 'conference', 'seminar', 'workshop', 'education', 
                                'learning', 'certification', 'qualification'],
    'Transport & Travel': ['fuel', 'gasoline', 'diesel', 'car', 'vehicle', 'maintenance', 'repair', 'taxi', 'uber', 
                          'lyft', 'transport', 'travel', 'flight', 'train', 'bus', 'subway', 'metro', 'parking'],
    'Professional Services': ['lawyer', 'legal', 'accountant', 'accounting', 'consultant', 'consulting', 'advisor',
                             'service', 'professional', 'contractor', 'outsourcing', 'recruiter'],
    'Marketing & Business Development': ['advertising', 'marketing', 'promotion', 'campaign', 'seo', 'sem', 'social media',
                                        'branding', 'design', 'website', 'domain', 'hosting', 'business cards', 'flyers'],
    # 50% coefficient category (50% deductible - medium deduction rate)
    'Meals & Entertainment': ['restaurant', 'cafe', 'coffee', 'lunch', 'dinner', 'breakfast', 'food', 'meal', 
                             'catering', 'delivery', 'entertainment', 'client meeting'],
    # 95% coefficient category (5% deductible - low deduction rate)
    'Social Security Contributions': ['social security', 'contribuições', 'segurança social'],
    'Insurance': ['insurance', 'coverage', 'policy', 'premium', 'health', 'liability', 'property'],
    'Banking & Financial': ['bank', 'banking', 'finance', 'financial', 'fee', 'interest', 'loan', 'credit', 'transaction'],
    # Non-deductible or special considerations
    'Personal Expenses': ['personal', 'private', 'clothing', 'groceries', 'household', 'non-business', 'entertainment'],
    'Miscellaneous': [] # Default category
}

class ExpenseProcessor:
    def __init__(self, folder_path: str, output_file: str = 'expenses.csv', use_openai: bool = False, openai_api_key: str = None, portugal_tax_optimization: bool = False, recursive: bool = False):
        self.folder_path = folder_path
        self.output_file = output_file
        self.exchange_rates = {}
        self.expenses = []
        self.unprocessed_files = []  # New: Track files that couldn't be processed
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.portugal_tax_optimization = portugal_tax_optimization
        self.recursive = recursive  # Store the recursive flag
        
    def process_all_files(self):
        """Process all expense files in the specified folder"""
        # Get all files with supported extensions
        file_patterns = ['*.pdf', '*.jpg', '*.jpeg', '*.png']
        files = []
        
        # Determine whether to search recursively
        if self.recursive:
            for pattern in file_patterns:
                for root, _, _ in os.walk(self.folder_path):
                    files.extend(glob.glob(os.path.join(root, pattern)))
        else:
            for pattern in file_patterns:
                files.extend(glob.glob(os.path.join(self.folder_path, pattern)))
        
        logger.info(f"Found {len(files)} files to process")
        
        # Get latest exchange rates
        self.fetch_exchange_rates()
        
        # Process each file
        for file_path in files:
            try:
                logger.info(f"Processing file: {file_path}")
                text = self.extract_text(file_path)
                if text:
                    expense_data = self.parse_expense_data(text, file_path)
                    if expense_data:
                        self.expenses.append(expense_data)
                    else:
                        # New: Track unprocessed files with partial data
                        partial_data = self.extract_partial_data(text, file_path)
                        partial_data['reason'] = 'Missing required data'
                        self.unprocessed_files.append(partial_data)
                else:
                    # New: Track files with text extraction issues
                    self.unprocessed_files.append({
                        'file': os.path.basename(file_path),
                        'reason': 'Could not extract text',
                        'extracted_text': ''
                    })
                    logger.warning(f"Could not extract text from {file_path}")
            except Exception as e:
                # New: Track files with processing errors
                self.unprocessed_files.append({
                    'file': os.path.basename(file_path),
                    'reason': f'Error: {str(e)}',
                    'extracted_text': text if 'text' in locals() else ''
                })
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Save results to CSV
        self.save_to_csv()
        logger.info(f"Processed {len(self.expenses)} expenses and saved to {self.output_file}")
        
        # New: Save unprocessed files to a separate CSV
        if self.unprocessed_files:
            self.save_unprocessed_to_csv()
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from the file based on its type"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return self.extract_text_from_image(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # First try direct text extraction with PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
        
        # If text extraction didn't yield much, try OCR
        if len(text.strip()) < 100:
            logger.info(f"PDF had little text content, trying OCR: {pdf_path}")
            try:
                # Convert PDF to images
                images = convert_from_path(pdf_path)
                
                # Perform OCR on each image
                for i, image in enumerate(images):
                    image_text = pytesseract.image_to_string(image)
                    text += image_text + "\n"
            except Exception as e:
                logger.error(f"Error in PDF OCR processing: {str(e)}")
        
        return text
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image file using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error in image OCR processing: {str(e)}")
            return ""
    
    def fetch_exchange_rates(self):
        """Fetch current exchange rates from API"""
        # Try to get API key from environment variables
        api_key = os.environ.get('EXCHANGE_RATE_API_KEY')
        
        try:
            # Use API key if available, otherwise use the free endpoint
            if api_key:
                url = f'https://v6.exchangerate-api.com/v6/{api_key}/latest/EUR'
                logger.info("Using Exchange Rate API with authentication")
            else:
                url = 'https://api.exchangerate-api.com/v4/latest/EUR'
                logger.info("Using free Exchange Rate API (limited requests)")
                
            response = requests.get(url)
            data = response.json()
            
            # Handle different API response formats
            if 'rates' in data:
                rates = data['rates']
            elif 'conversion_rates' in data:  # v6 API format
                rates = data['conversion_rates']
            else:
                raise ValueError("Unexpected API response format")
            
            # Invert rates since we want to convert TO EUR
            eur_rate = rates['EUR']  # This should be 1.0
            usd_rate = rates['USD']
            gbp_rate = rates['GBP']
            
            # Add ISK rate if available
            isk_rate = rates.get('ISK', 150.0)  # Default to a reasonable ISK rate if not available
            
            self.exchange_rates = {
                'EUR': 1.0,
                'USD': eur_rate / usd_rate,
                'GBP': eur_rate / gbp_rate,
                'ISK': eur_rate / isk_rate
            }
            
            logger.info(f"Exchange rates: EUR=1.0, USD={self.exchange_rates['USD']:.4f}, GBP={self.exchange_rates['GBP']:.4f}, ISK={self.exchange_rates['ISK']:.6f}")
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {str(e)}")
            # Fallback to some default rates
            self.exchange_rates = {
                'EUR': 1.0,
                'USD': 0.85,
                'GBP': 1.15,
                'ISK': 0.0067  # Approximate EUR/ISK rate
            }
            logger.warning("Using fallback exchange rates")
    
    def parse_expense_data(self, text: str, file_path: str) -> Optional[Dict]:
        """Extract expense data from the text"""
        # Convert to lowercase for easier pattern matching
        text_lower = text.lower()
        
        # Initialize expense data
        expense = {
            'file': os.path.basename(file_path),
            'vendor': self.extract_vendor(text),
            'original_cost': 0.0,
            'original_currency': '',
            'converted_cost': 0.0,
            'date': '',
            'category': 'Miscellaneous'
        }
        
        # Extract amount and currency
        amount_info = self.extract_amount_and_currency(text)
        if amount_info:
            expense['original_cost'] = amount_info[0]
            expense['original_currency'] = amount_info[1]
            expense['converted_cost'] = self.convert_to_euro(amount_info[0], amount_info[1])
        
        # Extract date
        expense['date'] = self.extract_date(text)
        
        # Categorize expense
        expense['category'] = self.categorize_expense(text_lower, expense['vendor'].lower())
        
        # If using OpenAI and we're missing key data, try OpenAI extraction
        if self.use_openai and self.openai_api_key:
            try:
                ai_extracted = self.extract_with_openai(text, file_path)
                if ai_extracted:
                    # Always prioritize AI-extracted vendor name if available
                    if ai_extracted.get('vendor'):
                        expense['vendor'] = self.clean_vendor_name(ai_extracted['vendor'])
                    
                    # Update with AI-extracted data where our extraction failed
                    if expense['original_cost'] == 0 and ai_extracted.get('original_cost', 0) > 0:
                        expense['original_cost'] = ai_extracted['original_cost']
                        expense['original_currency'] = ai_extracted.get('original_currency', 'EUR')
                        expense['converted_cost'] = self.convert_to_euro(
                            ai_extracted['original_cost'], 
                            ai_extracted.get('original_currency', 'EUR')
                        )
                    if not expense['date'] and ai_extracted.get('date'):
                        expense['date'] = ai_extracted['date']
            except Exception as e:
                logger.warning(f"OpenAI extraction failed: {str(e)}")
        
        # Check if we have the minimum required information
        if expense['vendor'] and expense['original_cost'] > 0 and expense['original_currency']:
            return expense
        else:
            logger.warning(f"Couldn't extract required data from {file_path}")
            return None
    
    def extract_partial_data(self, text: str, file_path: str) -> Dict:
        """Extract whatever data we can, even if incomplete, for the unprocessed files log"""
        # This is similar to parse_expense_data but doesn't filter out incomplete data
        text_lower = text.lower()
        
        partial_data = {
            'file': os.path.basename(file_path),
            'vendor': self.extract_vendor(text) or '',
            'original_cost': 0.0,
            'original_currency': '',
            'date': self.extract_date(text) or '',
            'extracted_text': text[:500] + ('...' if len(text) > 500 else '')  # Include truncated text
        }
        
        # Extract amount and currency
        amount_info = self.extract_amount_and_currency(text)
        if amount_info:
            partial_data['original_cost'] = amount_info[0]
            partial_data['original_currency'] = amount_info[1]
        
        return partial_data
    
    def extract_with_openai(self, text: str, file_path: str) -> Optional[Dict]:
        """Use OpenAI to extract expense data from text"""
        # Skip if no API key or not enabled
        if not self.use_openai or not self.openai_api_key:
            return None
            
        import openai
        openai.api_key = self.openai_api_key
        
        # Truncate text if it's too long
        max_text_length = 4000
        extracted_text = text[:max_text_length] + ("..." if len(text) > max_text_length else "")
        
        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from invoices and receipts. Respond in JSON format only."},
                    {"role": "user", "content": f"""Extract the following information from this invoice/receipt text as JSON:
                    - vendor: The company or vendor name (IMPORTANT: Extract the actual business name, not labels like 'LLC', 'Bill', or 'Invoice')
                    - original_cost: The total amount (numeric only)
                    - original_currency: The currency code (EUR, USD, GBP, etc.)
                    - date: The invoice/receipt date in YYYY-MM-DD format
                    
                    If you cannot find a specific field, return null for that field.
                    Text: {extracted_text}"""}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            logger.info(f"OpenAI extraction successful for {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None
    
    def extract_vendor(self, text: str) -> str:
        """Extract vendor/company name from text"""
        # Look for common vendor name patterns (e.g., after "From:", "Supplier:", "Company:" etc.)
        vendor_patterns = [
            r'(?:vendor|supplier|company|business|from|issued by|billed from)\s*:\s*([A-Za-z0-9\s&.,]+)',
            r'(?:[Ii]nvoice from)\s*([A-Za-z0-9\s&.,]+)',
            r'^([A-Za-z0-9\s&.,]+)(?:\s*invoice)',
            # Add new patterns for restaurants and taxis
            r'(?:Atendido\s+Por|Processado\s+por).*?\n([A-Za-z0-9\s&.,]+)',  # Restaurant staff/processor
            r'(?:HREYFILL|TAXI).*?SIMI\s+(\d+.*\d+)'  # Taxi phone number as identifier
        ]
        
        for pattern in vendor_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                vendor = matches.group(1).strip()
                if len(vendor) > 3 and len(vendor) < 50:
                    return self.clean_vendor_name(vendor)
        
        # Special case for restaurant bills - often the restaurant name is at the top
        lines = text.strip().split('\n')
        for i in range(min(3, len(lines))):
            line = lines[i].strip()
            if 3 < len(line) < 40 and not re.search(r'invoice|receipt|document|statement|nota|fatura', line, re.IGNORECASE):
                return self.clean_vendor_name(line)
        
        # Look for taxi company names
        taxi_companies = ['HREYFILL TAXI', 'TAXI', 'CAB', 'UBER']
        for company in taxi_companies:
            if company in text.upper():
                return company.title()  # Return in title case
        
        # Look for restaurant identifiers
        restaurant_identifiers = ['RESTAURANT', 'RESTAURANTE', 'CAFE', 'BAR', 'FOOD', 'MEAL', 'VERSAILLES']
        for identifier in restaurant_identifiers:
            if identifier in text.upper():
                # Try to get the actual name by looking at context
                context_pattern = r'([A-Za-z0-9\s&.,]+)\s*(?:' + identifier + ')'
                matches = re.search(context_pattern, text, re.IGNORECASE)
                if matches:
                    vendor = matches.group(1).strip()
                    if len(vendor) > 2:
                        return self.clean_vendor_name(vendor + ' ' + identifier)
                else:
                    # Look at the first few lines for a potential name
                    for i in range(min(3, len(lines))):
                        if len(lines[i].strip()) > 3 and len(lines[i].strip()) < 30:
                            return self.clean_vendor_name(lines[i].strip())
                
                # If no specific name found, return the identifier
                return identifier.title()
        
        # If no vendor found with patterns, try to extract the most prominent company name
        # This is a simplified approach - look for words that appear prominently at the top of the document
        for i in range(min(5, len(lines))):
            line = lines[i].strip()
            if 3 < len(line) < 40 and not re.search(r'invoice|receipt|document|statement', line, re.IGNORECASE):
                words = re.findall(r'[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*', line)
                for word in words:
                    if len(word) > 3:
                        return self.clean_vendor_name(word)
        
        # Default to filename without extension if we can't find a vendor
        filename = os.path.basename(text.split('\n')[0] if '\n' in text else text)
        return self.clean_vendor_name(re.sub(r'\.\w+$', '', filename))
    
    def clean_vendor_name(self, vendor: str) -> str:
        """Clean and normalize vendor names"""
        # Convert to lowercase for comparison
        vendor_lower = vendor.lower().strip()
        
        # Define known vendor name mappings (incorrect -> correct)
        vendor_mappings = {
            'llc': 'OpenAI',
            '◊': 'Lovable',
            'bill': 'EE',
            'att': 'Starlink',
            'account raffingers llp': 'Raffingers',
            'account  raffingers llp': 'Raffingers',
            'invoice': 'Udemy',
            # New mappings for restaurant and taxi receipts
            'kuittun': 'Hreyfill Taxi',
            'hreyfill taxi': 'Hreyfill Taxi',
            'hreyfill': 'Hreyfill Taxi',
            'versailles a': 'Versailles Restaurant',
            'versailles': 'Versailles Restaurant',
            'food': 'Restaurant',
            'restaurante': 'Restaurant'
        }
        
        # Check for exact matches in our mapping
        if vendor_lower in vendor_mappings:
            return vendor_mappings[vendor_lower]
        
        # Check for partial matches
        for incorrect, correct in vendor_mappings.items():
            if incorrect in vendor_lower:
                return correct
        
        # Remove common prefixes/suffixes that aren't part of the actual vendor name
        vendor = re.sub(r'^(invoice|receipt|statement|bill|from|to|nota|fatura)\s+', '', vendor, flags=re.IGNORECASE)
        vendor = re.sub(r'\s+(invoice|receipt|statement|bill|number|account|llc|ltd|inc|co\.)\s*$', '', vendor, flags=re.IGNORECASE)
        
        # Handle special cases
        if 'reykjavik' in vendor_lower or 'taxi' in vendor_lower:
            return 'Hreyfill Taxi'
        
        if 'restaurant' in vendor_lower or 'cafe' in vendor_lower or 'bar' in vendor_lower:
            # Try to extract the actual restaurant name (usually comes before "Restaurant")
            parts = re.split(r'\s+(?:restaurant|cafe|bar)', vendor_lower, flags=re.IGNORECASE)
            if parts and len(parts[0]) > 2:
                return parts[0].title() + ' Restaurant'
            else:
                return 'Restaurant'
        
        # Normalize whitespace
        vendor = ' '.join(vendor.split())
        
        # Capitalize properly
        if len(vendor) > 0:
            return vendor.title()
        
        return vendor
    
    def extract_amount_and_currency(self, text: str) -> Optional[Tuple[float, str]]:
        """Extract amount and currency from text"""
        # Common currency symbols and codes
        currency_patterns = {
            'EUR': [r'€\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*€', r'EUR\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*EUR'],
            'USD': [r'\$\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*\$', r'USD\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*USD'],
            'GBP': [r'£\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*£', r'GBP\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*GBP'],
            'ISK': [r'ISK\s*(\d+[.,]\d+)', r'(\d+[.,]\d+)\s*ISK']  # Add support for Icelandic Krona
        }
        
        # Look for total amount patterns first
        total_patterns = [
            r'(?:total|amount|sum|due|pago em|total\s*:)\s*(?:[A-Z]{3}|\$|€|£)?\s*(\d+[.,]\d+)',
            r'(?:total|amount|sum|due|pago em|total\s*:)\s*(\d+[.,]\d+)\s*(?:[A-Z]{3}|\$|€|£)?',
            r'(?:total).*?(\d+[.,]\d+)',  # More relaxed pattern for restaurant bills
            r'(?:troco|cartao bancario).*?(\d+[.,]\d+)'  # Common in Portuguese receipts
        ]
        
        # First try to find amounts with currency indicators
        for currency, patterns in currency_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).replace(',', '.')
                    try:
                        return float(amount_str), currency
                    except ValueError:
                        continue
        
        # Try to find a total amount and guess currency
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '.')
                try:
                    amount = float(amount_str)
                    # Try to guess currency
                    if '€' in text or 'EUR' in text.upper():
                        return amount, 'EUR'
                    elif '$' in text or 'USD' in text.upper():
                        return amount, 'USD'
                    elif '£' in text or 'GBP' in text.upper():
                        return amount, 'GBP'
                    elif 'ISK' in text.upper() or 'REYKJAVIK' in text.upper():  # Context clues for Icelandic currency
                        return amount, 'ISK'
                    else:
                        # Default to EUR if we can't determine
                        return amount, 'EUR'
                except ValueError:
                    continue
        
        # Specific pattern for taxi receipts with format like "1,68 km./533,00"
        taxi_patterns = [
            r'(\d+[.,]\d+)\s*km\s*[./]\s*(\d+[.,]\d+)',  # Distance and price pattern
            r'[sS]tartgjald.*?(\d+[.,]\d+)',  # Start charge in taxi receipts
            r'total.*?(\d+[.,]\d+)'  # Final total
        ]
        
        for pattern in taxi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # If it's the distance/price pattern, use the second group (price)
                if pattern == r'(\d+[.,]\d+)\s*km\s*[./]\s*(\d+[.,]\d+)':
                    amount_str = match.group(2).replace(',', '.')
                else:
                    amount_str = match.group(1).replace(',', '.')
                
                try:
                    amount = float(amount_str)
                    # For Icelandic taxi receipts
                    if 'REYKJAVIK' in text.upper() or 'HREYFILL' in text.upper():
                        return amount, 'ISK'
                    else:
                        return amount, 'EUR'  # Default to EUR for taxi receipts
                except ValueError:
                    continue
        
        # Restaurant bill specific patterns
        restaurant_patterns = [
            r'(?:total|pago).*?(\d+[.,]\d+)\s*€?',  # Total paid
            r'(?:cartao bancario|visa|mastercard|amex).*?(\d+[.,]\d+)',  # Card payment
            r'(?:iva|base).*?(\d+[.,]\d+)'  # Tax base amount which can help identify the total
        ]
        
        for pattern in restaurant_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '.')
                try:
                    amount = float(amount_str)
                    # If in Portugal or mentions Euro
                    if 'PORTUGAL' in text.upper() or 'VERSAILLES' in text.upper() or '€' in text:
                        return amount, 'EUR'
                    else:
                        return amount, 'EUR'  # Default to EUR
                except ValueError:
                    continue
        
        return None
    
    def extract_date(self, text: str) -> str:
        """Extract date from text"""
        # Look for common date formats
        date_patterns = [
            # YYYY-MM-DD
            r'(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
            # DD-MM-YYYY
            r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
            # Month name formats
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            # Look for date labels
            r'(?:Date|Invoice Date|Transaction Date|Data):\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
            r'(?:Date|Invoice Date|Transaction Date|Data):\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
            # Portuguese date formats often found in receipts
            r'(?:data|hora).*?(\d{4}-\d{2}-\d{2})',
            r'(?:data|hora).*?(\d{2}-\d{2}-\d{4})',
            r'(?:data|hora|[dD]ata\s*:).*?(\d{2}/\d{2}/\d{4})',
            # Taxi receipt date formats
            r'STILT\).*?:?\s*(\d{2}[-/\.]\w{3}[-/\.]\d{4})',  # Format like 09-FEB-2022
            r'DABS.*?:?\s*(\d{2}[-/\.]\w{3}[-/\.]\d{4})'  # Another common format in taxi receipts
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                
                # Try different date parsing formats
                date_formats = [
                    '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
                    '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
                    '%d %b %Y', '%d %B %Y',
                    '%b %d %Y', '%b %d, %Y',
                    '%B %d %Y', '%B %d, %Y',
                    '%d-%b-%Y', '%d/%b/%Y', '%d.%b.%Y',  # For formats like 09-FEB-2022
                    '%d-%B-%Y', '%d/%B/%Y', '%d.%B.%Y'
                ]
                
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        return date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                
                # If standard parsing fails, try some special cases
                # For Portuguese format: DD-MM-YYYY or DD/MM/YYYY
                if re.match(r'\d{2}[-/\.]\d{2}[-/\.]\d{4}', date_str):
                    parts = re.split(r'[-/\.]', date_str)
                    if len(parts) == 3:
                        try:
                            return f"{parts[2]}-{parts[1]}-{parts[0]}"  # Convert to YYYY-MM-DD
                        except:
                            pass
                
                # For unusual formats like 09-FEB-2022
                month_map = {
                    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
                    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                }
                
                if re.match(r'\d{2}[-/\.][a-zA-Z]{3}[-/\.]\d{4}', date_str):
                    parts = re.split(r'[-/\.]', date_str)
                    if len(parts) == 3 and parts[1].lower()[:3] in month_map:
                        try:
                            day = parts[0]
                            month = month_map[parts[1].lower()[:3]]
                            year = parts[2]
                            return f"{year}-{month}-{day}"  # Convert to YYYY-MM-DD
                        except:
                            pass
        
        # If no date found, use today's date
        return datetime.now().strftime('%Y-%m-%d')
    
    def convert_to_euro(self, amount: float, currency: str) -> float:
        """Convert amount to EURO based on exchange rates"""
        rate = self.exchange_rates.get(currency.upper(), 1.0)
        return amount * rate
    
    def categorize_expense(self, text: str, vendor: str) -> str:
        """Categorize expense based on text content and vendor"""
        if self.use_openai and self.openai_api_key and self.portugal_tax_optimization:
            # Try using AI for Portugal tax optimization
            ai_category = self.categorize_with_openai_portugal(text, vendor)
            if ai_category:
                return ai_category
                
        # If AI categorization failed or not using AI, use the rules-based approach
        categories = PORTUGAL_TAX_CATEGORIES if self.portugal_tax_optimization else CATEGORIES
        
        # Check each category's keywords
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in text.lower() or keyword.lower() in vendor.lower():
                    return category
        
        # Default category
        return "Miscellaneous"
    
    def categorize_with_openai_portugal(self, text: str, vendor: str) -> str:
        """Categorize expense using OpenAI with Portuguese tax optimization"""
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            prompt_text = f"""As a Portuguese tax expert, categorize this expense for a self-employed freelancer in Portugal.

Expense text: {text[:500]}
Vendor: {vendor}

Choose the most tax-efficient category from the following options:

15% coefficient categories (75% deductible - best for deductions):
- Material Costs & Supplies: office supplies, consumables
- Equipment & Technology: hardware, software, subscriptions, tech services
- Workspace Expenses: rent, utilities, internet, telecommunications
- Professional Development: courses, books, conferences
- Transport & Travel: fuel, public transport, business travel
- Professional Services: accounting, legal, consulting services
- Marketing & Business Development: advertising, design, promotion

50% coefficient category (50% deductible):
- Meals & Entertainment: business meals, client entertainment

95% coefficient category (5% deductible - least beneficial):
- Social Security Contributions
- Insurance
- Banking & Financial: bank fees, financial services

Non-deductible:
- Personal Expenses

Return ONLY the category name, nothing else."""

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Portuguese tax expert specializing in freelancer taxation."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3,
                max_tokens=20
            )
            
            category = response.choices[0].message.content.strip()
            
            # Map the response to our categories
            # Check if the category directly matches one of our categories
            for our_category in PORTUGAL_TAX_CATEGORIES.keys():
                if our_category.lower() in category.lower():
                    return our_category
            
            # If no direct match, return the best guess based on the AI response
            if "material" in category.lower() or "supplies" in category.lower():
                return "Material Costs & Supplies"
            elif "equipment" in category.lower() or "technology" in category.lower() or "tech" in category.lower():
                return "Equipment & Technology"
            elif "workspace" in category.lower() or "office" in category.lower() or "rent" in category.lower():
                return "Workspace Expenses"
            elif "professional development" in category.lower() or "education" in category.lower() or "learning" in category.lower():
                return "Professional Development"
            elif "transport" in category.lower() or "travel" in category.lower():
                return "Transport & Travel"
            elif "professional services" in category.lower() or "consulting" in category.lower() or "service" in category.lower():
                return "Professional Services"
            elif "marketing" in category.lower() or "advertising" in category.lower() or "business development" in category.lower():
                return "Marketing & Business Development"
            elif "meal" in category.lower() or "entertainment" in category.lower() or "food" in category.lower():
                return "Meals & Entertainment"
            elif "social security" in category.lower():
                return "Social Security Contributions"
            elif "insurance" in category.lower():
                return "Insurance"
            elif "banking" in category.lower() or "financial" in category.lower() or "bank" in category.lower():
                return "Banking & Financial"
            elif "personal" in category.lower():
                return "Personal Expenses"
            
            return "Miscellaneous"
            
        except Exception as e:
            logger.warning(f"OpenAI categorization failed: {str(e)}")
            return ""
    
    def save_to_csv(self):
        """Save processed expense data to CSV file"""
        if not self.expenses:
            logger.warning("No expenses to save")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.expenses)
        
        # Reorder columns to match preferred format
        columns = ['vendor', 'original_cost', 'original_currency', 'converted_cost', 'date', 'category', 'file']
        df = df[columns]
        
        # Rename columns to match preferred format
        df.columns = ['Company/Vendor', 'Original Cost', 'Currency', 'Converted Cost (EUR)', 'Date', 'Category', 'Filename']
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        
        # Also create a summary by category
        summary = df.groupby('Category')['Converted Cost (EUR)'].agg(['sum', 'count'])
        summary.columns = ['Total (EUR)', 'Count']
        summary = summary.sort_values('Total (EUR)', ascending=False)
        summary_file = os.path.splitext(self.output_file)[0] + '_summary.csv'
        summary.to_csv(summary_file)
        logger.info(f"Category summary saved to {summary_file}")
    
    def save_unprocessed_to_csv(self):
        """Save information about unprocessed files to CSV"""
        if not self.unprocessed_files:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.unprocessed_files)
        
        # Save to CSV
        unprocessed_file = os.path.splitext(self.output_file)[0] + '_unprocessed.csv'
        df.to_csv(unprocessed_file, index=False)
        logger.info(f"Information about {len(self.unprocessed_files)} unprocessed files saved to {unprocessed_file}")

def main():
    """Main function to parse arguments and run the processor"""
    parser = argparse.ArgumentParser(description='Process expense files and generate a CSV report')
    parser.add_argument('folder', help='Folder containing expense files')
    parser.add_argument('--output', '-o', default='expenses.csv', help='Output CSV file name')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI to help extract data from difficult documents')
    parser.add_argument('--openai-api-key', help='OpenAI API key (required if --use-openai is specified)')
    parser.add_argument('--portugal-tax', action='store_true', help='Optimize categorization for Portuguese freelancer taxes')
    parser.add_argument('--recursive', action='store_true', help='Search for files recursively in subdirectories')
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.isdir(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        return
    
    # Check if OpenAI is requested but no API key provided
    if args.use_openai and not args.openai_api_key:
        # Try to get from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OpenAI API key is required when --use-openai is specified. "
                         "Provide it with --openai-api-key or set the OPENAI_API_KEY environment variable.")
            return
        args.openai_api_key = openai_api_key
    
    logger.info(f"Starting expense processing with options: output={args.output}, use_openai={args.use_openai}, "
               f"portugal_tax={args.portugal_tax}, recursive={args.recursive}")
    
    # Process expenses
    processor = ExpenseProcessor(
        args.folder, 
        args.output,
        use_openai=args.use_openai,
        openai_api_key=args.openai_api_key,
        portugal_tax_optimization=args.portugal_tax,
        recursive=args.recursive
    )
    processor.process_all_files()

if __name__ == "__main__":
    main()
