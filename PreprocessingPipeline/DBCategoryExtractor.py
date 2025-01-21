import os
import re

class CategoryExtractor:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
    
    def extract_category(self, filename, category):
        filepath = os.path.join(self.input_directory, filename)
        
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Generalisierte Suche basierend auf übergebener Kategorie
        pattern = rf'>\s*<{category}>\s*\n(.*?)\s*>\s*<'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            category_content = match.group(1).strip()
            return category_content
        
        return None
    
    def process_files(self, category):
        for filename in os.listdir(self.input_directory):
            if filename.endswith('.txt'):
                category_content = self.extract_category(filename, category)
                
                if category_content:
                    new_filename = f"{filename.replace('.txt', '')}.{category.lower()}.txt"
                    output_filepath = os.path.join(self.output_directory, new_filename)
                    
                    with open(output_filepath, 'w', encoding='utf-8') as output_file:
                        output_file.write(category_content)
                    
                    print(f"Extrahiert: {filename} -> {new_filename}")

# Beispielverwendung
if __name__ == "__main__":
    extractor = CategoryExtractor('data/temp/split_db', 'data/chemnames/unsorted')

    extractor.process_files('CHEMNAME')  # oder jede andere Kategorie