**federated_learning_implementation using mnist dataset**

**setup**

clone the respositery
  
        git clone "https://github.com/abrham17/federated_learning_implementation"

navigate to the right folder
        
        cd federated_learning_implementation

create and activate virtual environemnt

        pip install virtualenv
        python -m venv .venv
        source .venv/bin/activate
  
installing needed libraries

        pip install tensorflow-federated==0.74.0
        pip install django

navigate to django folder
        
        cd fl_learning

migrate models 

        python manage.py makemigrations
        python manage.py migrate

run the server

      python manage.py runserver    


**folder structure**

fl_learning/                     
    ├── fl_dashboard/  
    │     ├── migrations/               
    │     ├── templates/               
    │     │   └── dashboard.html
    │     ├──__init__.py
    │     ├── admin.py                  
    │     ├── apps.py                   
    │     ├── client.py                 
    │     ├── data_partition.py         
    │     ├── models.py                 
    │     ├── run_simulation.py        
    │     ├── server.py                 
    │     ├── tests.py                  
    │     ├── urls.py                   
    │     └── views.py                  
    ├── fl_learning/                
    │     ├──__init__.py
    │     ├──settings.py              
    │     ├── urls.py                   
    │     ├── asgi.py                   
    │     └── wsgi.py                   
    │
    ├── manage.py                     
