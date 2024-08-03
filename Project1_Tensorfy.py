#!/usr/bin/env python
# coding: utf-8

# In[2]:


import fnmatch
import os
def FileSearching(directory, searchvalue):
    """Searches for a file with the given name/extension in the specified directory.

    Args:
        directory: The directory to search in.
        searchvalue: The name/extension of the file to search for.

    Returns:
        A list of full paths of files that match the search criteria. If none are found, an empty list is returned.
    """
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatchcase(file, f"*{searchvalue}*") or file.endswith(f".{searchvalue}"):
                results.append(os.path.join(root, file))
    return results


 


import re

def rename_file(file_path, new_name_rule):
    """Renames a file based on a given rule.

    Args:
        file_path: The full path of the file to be renamed.
        new_name_rule: A string representing the new name rule or regular expression.

    Returns:
        True if the file was renamed successfully, False otherwise.
    """
    try:
        file_dir, file_name = os.path.split(file_path)
        new_name = new_name_rule.format(file_name=file_name)  # Basic formatting new_{file.txt}.txt
        new_name = re.sub(r'\{.*\}', lambda m: m.group()[1:-1], new_name)  # Regular expression substitution
        new_file_path = os.path.join(file_dir, new_name)
        os.rename(file_path, new_file_path)
        print('Done renaming')
        return True
    except Exception as e:
        print(f"Error renaming file: {e}")
        return False

# In[ ]:


 


# In[ ]:


import tempfile #for creation of temp file
import shutil #high level operation on file

import tempfile
import shutil

def modify_file_content(file_path, find_text, replace_text):
    """
    Modifies the content of a file based on provided text to find and replace.

    Args:
        file_path: The full path of the file to be modified.
        find_text: The text to find in the file content.
        replace_text: The text to replace the found text with.

    Returns:
        A tuple (success, modified_content) where:
            - success (bool) is True if the file was modified successfully, False otherwise.
            - modified_content (str) is the content of the file after modification, or None if not successful.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
         
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
         
        modified_content = content.replace(find_text, replace_text)
        
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp()
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            temp_file.write(modified_content)

         
        shutil.move(temp_path, file_path)

        print(f"File modified successfully: {file_path}")
        
    except Exception as e:
        print(f"Error modifying file: {e}")
        return False


 
 


from pathlib import Path

def file_operation(sources, dest, operation='copy'):
  """
  Performs file or directory operations (copy or move) between multiple sources and a destination.

  Args:
    sources: A list of source file or directory paths.
    dest: The destination directory path.
    operation: The operation to perform ('copy' or 'move'). Defaults to 'copy'.
  """

  for source in sources:
    source_path = Path(source)
    dest_path = Path(dest)

    if source_path.is_file():
        if dest_path.is_dir():
                dest_path = dest_path / source_path.name  # Create destination file path
        if operation == 'copy':
                shutil.copy(source, dest)
                print(f"Copied {source} to {dest}")
        elif operation == 'move':
                shutil.move(source, dest)
                print(f"Moved {source} to {dest}")
        else:
               raise ValueError("Invalid operation. Choose 'copy' or 'move'.")
    elif source_path.is_dir():
        if dest_path.is_file():
            raise ValueError("Destination must be a directory when copying or moving a directory.")
        if operation == 'copy':
            shutil.copytree(source, dest, dirs_exist_ok=True)
            print(f"Copied directory {source} to {dest}")
        elif operation == 'move':
            shutil.move(source, dest)
            print(f"Moved directory {source} to {dest}")
        else:
            raise ValueError("Invalid operation. Choose 'copy' or 'move'.")
    else:
        raise ValueError(f"Invalid source path: {source}")

 



# In[ ]:


import os
import shutil
import argparse
 
parser = argparse.ArgumentParser(description='File operations tool')
subparsers = parser.add_subparsers(dest='command', help='The command to execute')

    # Search command
search_parser = subparsers.add_parser('search', help='Search for files')
search_parser.add_argument('-d', '--directory', required=True, help='Directory to search in')
search_parser.add_argument('-s', '--searchvalue', required=True, help='File name or extension to search for')

    # Rename command
rename_parser = subparsers.add_parser('rename', help='Rename a file')
rename_parser.add_argument('-f', '--file_path', required=True, help='File path to rename')
rename_parser.add_argument('-n', '--new_name_rule', required=True, help='New name or rule')

    # Modify command
modify_parser = subparsers.add_parser('modify', help='Modify file content')
modify_parser.add_argument('-f', '--file_path', required=True, help='File path to modify')
modify_parser.add_argument('-find', '--find_text', required=True, help='Text to find')
modify_parser.add_argument('-replace', '--replace_text', required=True, help='Text to replace with')

    # Operation command
operation_parser = subparsers.add_parser('operation', help='Perform file operations (copy or move)')
operation_parser.add_argument('-sources', nargs='+', required=True, help='Source file or directory paths')
operation_parser.add_argument('-dest', required=True, help='Destination directory path')
operation_parser.add_argument('-op', '--operation', choices=['copy', 'move'], default='copy', help='Operation to perform (copy or move)')

args = parser.parse_args()

if args.command == 'search':
    results = FileSearching(args.directory, args.searchvalue)
    if results:
        print("Files found:")
        for result in results:
            print(result)

elif args.command == 'rename':
    rename_file(args.file_path, args.new_name_rule)

elif args.command == 'modify':
    modify_file_content(args.file_path, args.find_text, args.replace_text)

elif args.command == 'operation':
    file_operation(args.sources, args.dest, args.operation)

 


# In[26]:





# In[19]:


 

# In[37]:




