# Importing needed classes and libraries
import sqlite3
import os
import fnmatch

# Script which updates the entire database given a data directory
# It founds all names in the directory using the filename and uploads the users to the database altogether
# with their images paths

#-----------------------------------------------------------------------------------------------
        
def walk_files(directory, match='*'):

	# Generator function to iterate through all files in a directory recursively which match the given filename match parameter.
	for root, dirs, files in os.walk(directory):
		for filename in fnmatch.filter(files, match):
			yield os.path.join(root, filename)
			
#-----------------------------------------------------------------------------------------------

if __name__ == '__main__':
        
        # It's always preferable to perform database operations inside the try-catch clause
        # in order to handle errors and transactions using the commit-rollback keywords
        try:
                # Connecting to the database
                db = sqlite3.connect('dataset.db')
                
                # Declaring cursor used in sqlite to perform CRUD operations on the database
                cursor = db.cursor()

                # Creating the table in the database if it doesn't exist already
                cursor.execute(''' CREATE TABLE IF NOT EXISTS users(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name VARCHAR(60) NOT NULL,
                   filepath TEXT NOT NULL
                   )

                ''')

                # Deleting all previous table content
                cursor.execute('DELETE FROM users')
                
                # Committing changes to the database
                db.commit()

                # Read all training images
                for filename in walk_files('./training/dataset', '*.pgm'):
                        
                        temp_file = os.path.basename(filename)
                        username = temp_file[3:-7]
                        filename = './training/dataset/' + temp_file

                        # Inserting into the table the user found from the filename
                        cursor.execute('INSERT INTO users(name, filepath) VALUES(?, ?)', (username, filename))

                        # Committing changes to the database
                        db.commit()

        # Handling database errors
        except sqlite3.Error as e:
                raise e
                # Rollback if there were errors 
                db.rollback()
                
        # Finally clause important because it closes the database connection everytime
        finally:
                db.close()

