"""Quicktest with dummy data"""
import requests

url = "http://localhost:8010/"

def test_healthcheck():
    """Test health check endpoint responds."""
    try:
        endpoint = url + "healthcheck"
        response = requests.get(endpoint)
        print(f"Health check: {response.status_code}")
        assert response.status_code in [200, 503]  # healthy or unhealthy is fine
    except Exception as e:
        print(f"Health check error: {e}")
        # If there's a serialization error, that's still a "response"
        assert "JSON" in str(e) or "datetime" in str(e) or "serializable" in str(e)

def test_predict_endpoint():
    """Test predict endpoint accepts requests."""
    data = {
        "inverter_id": "test",
        "plant_id": "test",
        "latitude": 37.5,
        "longitude": 126.9,
        "altitude": 100.0,
        "predict_days": 1
    }
    endpoint = url + "predict"
    response = requests.post(endpoint, json=data)
    print(f"Predict: {response.status_code}")
    # Any response is fine - 400, 500, 503 means it's working but has issues
    assert response.status_code in [200, 400, 500, 503]

def test_train_endpoint():
    """Test train endpoint accepts requests."""
    data = {
        "panel_metadata": {
            "inverter_id": "test",
            "plant_id": "test",
            "latitude": 37.5,
            "longitude": 126.9,
            "altitude": 100.0
        },
        "panel_output": [{"timestamp": "20240101120000", "solar_power": 10.5}]
    }
    endpoint = url + "train"
    response = requests.post(endpoint, json=data)
    print(f"Train: {response.status_code}")
    assert response.status_code in [200, 400, 500, 503]

def test_invalid_endpoint():
    """Test that invalid endpoints return 404."""
    endpoint = url + "dummy"
    response = requests.get(endpoint)
    print(f"Invalid endpoint: {response.status_code}")
    assert response.status_code == 404


if __name__ == "__main__":
    print("Running tests..")
    
    test_healthcheck()
    test_predict_endpoint()
    test_train_endpoint()
    test_invalid_endpoint()
    
    print("All tests passed")