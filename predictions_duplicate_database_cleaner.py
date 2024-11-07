from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
from appwrite.query import Query
from dotenv import load_dotenv
import os
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Initialize Appwrite client
client = Client()
appwrite_endpoint = os.getenv('APPWRITE_ENDPOINT')
appwrite_project_id = os.getenv('APPWRITE_PROJECT_ID')
appwrite_api_key = os.getenv('APPWRITE_API_KEY')

if not all([appwrite_endpoint, appwrite_project_id, appwrite_api_key]):
    raise Exception("One or more Appwrite environment variables are not set")

client.set_endpoint(appwrite_endpoint)
client.set_project(appwrite_project_id)
client.set_key(appwrite_api_key)

databases = Databases(client)
database_id = os.getenv('APPWRITE_DATABASE_ID')

# Ensure that database ID is set
if not database_id:
    raise Exception("APPWRITE_DATABASE_ID environment variable not set")

predictions_collection_id = 'predictions'  # Collection for storing predictions

# Function to delete duplicate predictions based on prediction_time
def delete_duplicate_predictions():
    try:
        all_documents = []
        limit = 100  # Adjust as needed
        offset = 0
        total = None

        # Fetch all documents in the collection with pagination
        while True:
            response = databases.list_documents(
                database_id=database_id,
                collection_id=predictions_collection_id,
                queries=[
                    Query.limit(limit),
                    Query.offset(offset),
                    Query.order_asc('prediction_time')
                ]
            )

            documents = response['documents']
            total = response['total']

            all_documents.extend(documents)

            offset += limit

            if offset >= total:
                break

        print(f"Total documents fetched: {len(all_documents)}")

        # Build a dictionary to find duplicates based on prediction_time
        prediction_time_dict = defaultdict(list)

        for doc in all_documents:
            prediction_time = doc.get('prediction_time')
            if prediction_time:
                prediction_time_dict[prediction_time].append(doc)

        # Find duplicates and delete them
        duplicates_found = 0
        documents_deleted = 0

        for prediction_time, docs in prediction_time_dict.items():
            if len(docs) > 1:
                duplicates_found += 1
                # Keep the first document (or you can implement custom logic)
                docs_to_delete = docs[1:]  # All except the first one

                for doc in docs_to_delete:
                    document_id = doc['$id']
                    try:
                        databases.delete_document(
                            database_id=database_id,
                            collection_id=predictions_collection_id,
                            document_id=document_id
                        )
                        documents_deleted += 1
                        print(f"Deleted duplicate document ID: {document_id} with prediction_time: {prediction_time}")
                    except AppwriteException as e:
                        print(f"Error deleting document ID: {document_id}. Error: {e}")

        print(f"Total duplicates found: {duplicates_found}")
        print(f"Total documents deleted: {documents_deleted}")

    except AppwriteException as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    delete_duplicate_predictions()
