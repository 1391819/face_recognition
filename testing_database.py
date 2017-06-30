# Importing needed classes and libraries
import sqlite3

# Small script used to test the content of the database
#--------------------------------------------------------------

# Connecting to the database
db = sqlite3.connect('dataset.db')

# Declaring the cursor used to perform CRUD operations on the database
cursor = db.cursor()

# Executing query
cursor.execute('SELECT id, name, filepath FROM users')

# For each row returned from the query
for row in cursor:
    
    # Return the columns content inside the database
    print('{0} | {1} | {2}'.format(row[0], row[1], row[2]))

# Closing the database connection
db.close()


	    


	
