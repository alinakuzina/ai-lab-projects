import pytest
from flask import session
from app import app, db, User, Task

# Test Setup
@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.drop_all()

# Helper function to register a user
def register_user(client, username, password):
    return client.post('/register', data={
        'username': username,
        'password': password
    }, follow_redirects=True)

# Helper function to login a user
def login_user(client, username, password):
    return client.post('/login', data={
        'username': username,
        'password': password
    }, follow_redirects=True)

# Helper function to logout a user
def logout_user(client):
    return client.get('/logout', follow_redirects=True)

# Tests
def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the Task Manager!" in response.data

def test_about_page(client):
    response = client.get('/about')
    assert response.status_code == 200
    assert b"This is a task manager web app integrated with a chatbot." in response.data

def test_user_registration(client):
    response = register_user(client, "testuser", "testpassword")
    assert response.status_code == 200
    assert User.query.filter_by(username="testuser").first() is not None

def test_user_login(client):
    register_user(client, "testuser", "testpassword")
    response = login_user(client, "testuser", "testpassword")
    assert response.status_code == 200
    with client.session_transaction() as session_data:
        assert 'user_id' in session_data

def test_user_login_invalid(client):
    response = login_user(client, "invaliduser", "invalidpassword")
    assert response.status_code == 200
    assert b"Invalid credentials!" in response.data

def test_user_logout(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    response = logout_user(client)
    assert response.status_code == 200
    with client.session_transaction() as session_data:
        assert 'user_id' not in session_data

def test_task_creation(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    response = client.post('/tasks', data={
        'title': "Test Task",
        'description': "This is a test task."
    }, follow_redirects=True)
    assert response.status_code == 200
    assert Task.query.filter_by(title="Test Task").first() is not None

def test_task_edit(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    client.post('/tasks', data={
        'title': "Test Task",
        'description': "This is a test task."
    }, follow_redirects=True)
    task = Task.query.filter_by(title="Test Task").first()
    response = client.post(f'/task/{task.id}/edit', data={
        'title': "Updated Task",
        'description': "Updated description."
    }, follow_redirects=True)
    assert response.status_code == 200
    updated_task = Task.query.get(task.id)
    assert updated_task.title == "Updated Task"
    assert updated_task.description == "Updated description."

def test_task_delete(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    client.post('/tasks', data={
        'title': "Test Task",
        'description': "This is a test task."
    }, follow_redirects=True)
    task = Task.query.filter_by(title="Test Task").first()
    response = client.post(f'/task/{task.id}/delete', follow_redirects=True)
    assert response.status_code == 200
    assert Task.query.get(task.id) is None

def test_api_get_tasks(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    client.post('/tasks', data={
        'title': "Test Task",
        'description': "This is a test task."
    }, follow_redirects=True)
    response = client.get('/api/tasks')
    assert response.status_code == 200
    tasks = response.get_json()['tasks']
    assert len(tasks) == 1
    assert tasks[0]['title'] == "Test Task"

def test_api_create_task(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    response = client.post('/api/tasks', json={
        'title': "API Task",
        'description': "Task created via API"
    })
    assert response.status_code == 201
    task = Task.query.filter_by(title="API Task").first()
    assert task is not None

def test_api_update_task(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    client.post('/tasks', data={
        'title': "Test Task",
        'description': "This is a test task."
    }, follow_redirects=True)
    task = Task.query.filter_by(title="Test Task").first()
    response = client.put(f'/api/task/{task.id}', json={
        'title': "Updated API Task",
        'description': "Updated description via API"
    })
    assert response.status_code == 200
    updated_task = Task.query.get(task.id)
    assert updated_task.title == "Updated API Task"

def test_api_delete_task(client):
    register_user(client, "testuser", "testpassword")
    login_user(client, "testuser", "testpassword")
    client.post('/tasks', data={
        'title': "Test Task",
        'description': "This is a test task."
    }, follow_redirects=True)
    task = Task.query.filter_by(title="Test Task").first()
    response = client.delete(f'/api/task/{task.id}')
    assert response.status_code == 200
    assert Task.query.get(task.id) is None
