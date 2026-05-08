import os
from pathlib import Path

os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///./test_backend.db')
os.environ.setdefault('SYNC_DATABASE_URL', 'sqlite:///./test_backend.db')
os.environ.setdefault('AUTO_CREATE_SCHEMA', 'true')
os.environ.setdefault('SEED_REFERENCE_CATALOG', 'false')
os.environ.setdefault('INITIAL_ADMIN_EMAIL', 'admin@test.local')
os.environ.setdefault('INITIAL_ADMIN_PASSWORD', 'ChangeMe123!')

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoints():
    with TestClient(app) as client:
        assert client.get('/health/live').status_code == 200
        assert client.get('/health/ready').status_code == 200


def test_register_login_and_protected_profile_flow():
    db_path = Path('test_backend.db')
    if db_path.exists():
        db_path.unlink()

    with TestClient(app) as client:
        register_response = client.post(
            '/api/v1/auth/register',
            json={
                'email': 'doctor@example.com',
                'full_name': 'Doctor Demo',
                'password': 'SecurePass123!',
                'role': 'doctor',
            },
        )
        assert register_response.status_code == 200

        login_response = client.post(
            '/api/v1/auth/login',
            json={'email': 'doctor@example.com', 'password': 'SecurePass123!'},
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        assert 'access_token' in tokens

        me_response = client.get(
            '/api/v1/auth/me',
            headers={'Authorization': f"Bearer {tokens['access_token']}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()['email'] == 'doctor@example.com'
