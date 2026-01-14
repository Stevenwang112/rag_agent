
import os
import time
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_SERVICE_KEY')
supabase = create_client(url, key)

def update_metadata():
    print("🚀 Starting Metadata Update (Adding 'company_name')...")
    
    # 1. Fetch all parents to map ID -> Source
    # We assume reasonable number of docs. For production, paginate this.
    try:
        response = supabase.table('document_parents').select('id, metadata').execute()
        parents = response.data
    except Exception as e:
        print(f"Error fetching parents: {e}")
        return

    print(f"  Found {len(parents)} parent documents.")
    
    updates_count = 0
    
    for p in parents:
        pid = p['id']
        meta = p.get('metadata', {})
        source = meta.get('source', '')
        
        company = None
        if "ES9" in source:
            company = "ES9"
        elif "SU7" in source:
            company = "SU7"
            
        if not company:
            print(f"  Skipping Parent {pid}: Source '{source}' unknown.")
            continue
            
        # 2. Update all chunks for this parent
        # We need to fetch chunks first to preserve existing metadata? 
        # Or does supabase merge? jsonb update usually replaces if not careful.
        # Safer: Fetch chunks, update dict, write back.
        
        try:
            chunks_resp = supabase.table('document_chunks').select('id, metadata').eq('parent_id', pid).execute()
            chunks = chunks_resp.data
        except Exception as e:
            print(f"  Error fetching chunks for parent {pid}: {e}")
            continue
            
        if not chunks:
            continue
            
        # Prepare batch updates? Supabase-py upsert is easiest.
        chunk_updates = []
        for c in chunks:
            c_meta = c['metadata']
            if c_meta.get('company_name') == company:
                continue # Already done
            
            c_meta['company_name'] = company
            chunk_updates.append({
                "id": c['id'],
                "metadata": c_meta
            })
        
        if chunk_updates:
            try:
                # Upsert by ID to update metadata
                # Note: We must ensure we don't overwrite other fields. 
                # Since we pulled ID and Metadata, and we are upserting just those columns? No, Supabase upsert requires all columns or it might NULL others if not careful? 
                # Actually, `document_chunks` has required fields. Upserting partial data might be risky if we don't supply embedding/content etc?
                # Wait, Supabase `update` takes a filter.
                
                # Better approach: 
                # update document_chunks set metadata = metadata || '{"company_name": "ES9"}' where parent_id = ...
                # But treating Supabase as python client.
                # Let's try direct update execution on the parent_id group to save API calls!
                
                # UPDATE document_chunks SET metadata = jsonb_set(metadata, '{company_name}', '"ES9"') WHERE parent_id = '...'
                # But with python client...
                
                # Let's stick to the "Fetch -> Modify -> Upsert" but we need CONTENT and EMBEDDING... 
                # Ah, doing bulk update via client is hard if we don't have full row.
                
                # OPTIMIZED WAY: Use the python client's `update` method.
                # We can't do "jsonb merge" easily via standard client unless we fetch everything.
                # BUT, wait! We can just fetch all chunks with full data? "select *" might be heavy (embeddings).
                
                # Let's try iterating.
                print(f"  Updating {len(chunks)} chunks for Parent {pid} ({company})...")
                
                for c in chunks:
                   new_meta = c['metadata']
                   new_meta['company_name'] = company
                   # Update specifically this row's metadata
                   supabase.table('document_chunks').update({'metadata': new_meta}).eq('id', c['id']).execute()
                   updates_count += 1
                   
            except Exception as e:
                print(f"  Error updating chunks for parent {pid}: {e}")

    print(f"✅ Update Complete. Modified {updates_count} chunks.")

if __name__ == "__main__":
    update_metadata()
