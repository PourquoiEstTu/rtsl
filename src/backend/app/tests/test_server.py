import pytest
import httpx

SERVER_WSS = "https://rtsl.cas.mcmaster.ca:8000/"

# These test cases must be run locally connected to McMaster VPN as the server is on campus
@pytest.mark.asyncio
async def test_root_live_connection():
    # We use an async client to connect to the real IP
    async with httpx.AsyncClient(base_url=SERVER_WSS) as client:
        response = await client.get("/")
        
        # Verify the actual server response
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}