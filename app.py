from flask import Flask, render_template, url_for, redirect, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt 
import pandas as pd
# import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction. text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score


product = pd.read_csv("product.csv")
behavior = pd.read_csv("behavior.csv")
# product["image"] = product["image"].dropna()
username_list = behavior['username'].unique().tolist()
# product = product.sample(frac=1).reset_index(drop=True)
product_descriptions = product[["product_id", "description"]]
product_descriptions = product_descriptions.dropna()
product_descriptions = product_descriptions.head(500)
# Define a Tf-IDF Vectorizer Object. Remove all english stop words
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
product["description"] = product["description"].fillna("")

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(product["description"])
        
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Construct a reverse map of indices and product names
indices = pd.Series(product.index, index=product["title"])

def product_recommendations(product_name, cos=cosine_sim):
    for i in product.loc[product["title"] == product_name]["product_id"].values:
        product_id = i
    feature = []
    # Get the index of the product that matches the product name
    idx = indices[product_name]

    # Get the pairwise similarity scores
    # Enumerate adds a counter to the iterable and lets it be converted into a list of tuples
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    # Reverse gives us the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar products
    sim_scores = sim_scores[1:21]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar products
    # iloc allows us to retrieve rows from a data frame
    for i in product_indices:
        feature = product[["title","price","image"]].loc[product_indices].values.tolist()
    
    dictionary = dict(zip(product_indices, feature))
    return dictionary

def price_recommendations(lis, min_price, max_price, cosine_sim=cosine_sim):
    price_rec = []
    for product_name in lis:
        feature = []
        index = []
        # Get the index of the product that matches the product name
        idx = indices[product_name]

        # Get the pairwise similarity scores
        # Enumerate adds a counter to the iterable and lets it be converted into a list of tuples
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the products based on the similarity scores
        # Reverse gives us the similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar products
        sim_scores = sim_scores[1:10]

        # Get the product indices
        product_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar products
        # iloc allows us to retrieve rows from a data frame
        for i in product_indices:
            feature = product[["title","price","image"]].loc[product_indices].values.tolist()

        dictionary = dict(zip(product_indices, feature))
        count = 0
        price_dic = {}
        for key, value in dictionary.items():
            if count == 2:
                break
            else:
                if value[1] >= min_price and value[1] <= max_price:
                    price_dic[key] = value
                    count += 1
        price_rec.append(price_dic)
    return price_rec

def keyword_rec(keyword):
    count = 0
    top_10 = {}
    related_items = {}
    X = tfidf.fit_transform(product_descriptions["description"])
    # Top words in each cluster based on product descriptions
    true_k = 10
    # Fitting K-means to the dataset
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
    model.fit(X)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1][1:11]
    index = []
    feature = []
    Y = tfidf.transform([keyword])
    prediction = model.predict(Y)
    for k in prediction:
        for i in order_centroids[k]:
            index.append(i.item())
            feature.append(product[["title","price","image"]].loc[i].values.tolist())
    dictionary = dict(zip(index, feature))
    for key, value in dictionary.items():
        if keyword.title() in value[0]:
            if count == 10:
                break
            else:
                top_10[key] = value
                count += 1
    for key, value in dictionary.items():
        if keyword.title() not in value[0]:
            if count == 20:
                break
            else:
                related_items[key] = value
                count += 1    
    return [top_10,related_items]    


app = Flask(__name__)
# Database instance
db = SQLAlchemy(app)
bcrypt = Bcrypt(app) 
# Connect to Database
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
app.config['SECRET_KEY'] = 'thisisasecretkey'

# Loads user
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Loads the user object stored in the session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



# Table for database
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    # email = db.Column(db.String(100), nullable=False, unique=True)
    # Each user name is unique
    username = db.Column(db.String(20), nullable=False, unique=True)
    # Maximum hash is 80
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    # Different fields needed
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)])
    submit = SubmitField("Register")
    
    # To verify if username in user table
    def validate_username(self, username):
        # Queries the database table and checks if the username exists
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("This username already exists. Please choose a different one.")
        
class LoginForm(FlaskForm):
    # Different fields needed
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder':'username'})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder':'password'})
    submit = SubmitField("Login")



@app.route('/')
def home():
    return render_template("home.html")

@app.route('/login', endpoint='login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # To call the variable outside of the function and 
        # fetch the first instance of the user information from the database
        login.user=User.query.filter_by(username=form.username.data).first()
        if login.user:
            # Decrypt password hash
            if bcrypt.check_password_hash(login.user.password, form.password.data):
                login_user(login.user)
                if login.user.username in username_list:
                    return redirect(url_for('dashboard'))
                return redirect(url_for('shop_list'))
    return render_template("login.html", form=form)


items = []
@app.route('/shop_list', methods=['GET', 'POST'], endpoint='shop_list')
@login_required
def shop_list():
    if login.user is not None:
        shop_list.items=items
        item = request.form.get('content')
        if  item != None:
            shop_list.items.append(item)
    return render_template('shoplist.html', items=shop_list.items)

@app.route('/delete/<item>', methods=['GET', 'POST'], endpoint='delete')
@login_required
def delete(item):
    try: 
        shop_list.items.remove(item)
    
        return redirect('/shop_list')
    except:
        return 'There was an error deleting the task'

results={}
@app.route('/result', methods=['GET', 'POST'], endpoint='result')
@login_required
def result():
    global results
    # dic = None
    if login.user is not None:
        result= items
        for item in result:
            results[item]=(keyword_rec(item))
    return render_template('result.html', result=result, results=results)


@app.route('/search', endpoint='search', methods=['GET','POST'])
@login_required
def search():
    dictionary = dict()
    count = 0
    top_10 = {}
    related_items = {}
    if login.user is not None:
        keyword = request.form.get('keyword')
        # keyword_list.append(keyword)
        if keyword != None:
            X = tfidf.fit_transform(product_descriptions["description"])
            # Top words in each cluster based on product descriptions
            true_k = 10
            # Fitting K-means to the dataset
            model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
            model.fit(X)
            order_centroids = model.cluster_centers_.argsort()[:, ::-1][1:11]
            index = []
            feature = []
            Y = tfidf.transform([keyword])
            prediction = model.predict(Y)
            for k in prediction:
                for i in order_centroids[k]:
                    index.append(i.item())
                    feature.append(product[["title"]].loc[i].values.tolist())
            dictionary = dict(zip(index, feature)) 
            for key, value in dictionary.items():
                if keyword.title() in value[0]:
                    if count == 20:
                        break
                    else:
                        top_10[key] = value
                        count += 1
            for key, value in dictionary.items():
                if keyword.title() not in value[0]:
                    if count == 20:
                        break
                    else:
                        related_items[key] = value
                        count += 1
    return render_template("search.html", dictionary=[top_10, related_items], user=login.user.username, keyword=keyword, product=product)



behavior_list = []
purchase_list = []
view_list = []
rec_list = []

@app.route('/dashboard', endpoint='dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    min_price = 0
    max_price = 0
    # To access username
    if login.user is not None:
        product = pd.read_csv("product.csv")
        # product = product.sample(frac=1).reset_index(drop=True)
        dashboard.user = login.user.username
        for i in behavior[behavior["username"] == login.user.username].index.values:
            behavior_list.append(i)
        
        # Seperate the behaviours into lists
        for i in behavior_list:
            if behavior["behavior"].loc[i] == "buy":
                purchase_list.append(i)
            elif behavior["behavior"].loc[i] == "view":
                 view_list.append(i)
    
    
    # Price recommendations based on product name
        product_name_list = []
        new_purchase_list=[]
        price_list = []
        if len(purchase_list) >= 2 and len(purchase_list) <= 5:
            for i in purchase_list:
                price_list.append(behavior["price"].loc[i])
                new_purchase_list.append(behavior["product_name"].loc[i])
            min_price = min(price_list)
            max_price = max(price_list)
            for i in purchase_list:
                product_name_list.append(behavior["product_name"].loc[i])
        elif len(purchase_list) < 2:
            min_price = None
            max_price = None
            new_purchase_list = None
        else:
            new_purchase_list = purchase_list[-5:]
            for i in new_purchase_list:
                product_name_list.append(behavior["product_name"].loc[i])
                price_list.append(behavior["price"].loc[i])
            min_price = min(price_list)
            max_price = max(price_list)
            
        
        
        
        # min_product = behavior["product_name"].loc[min_price]
        # max_product = behavior["product_name"].loc[max_price]
        
        if new_purchase_list == None:
            price_rec_dic = None
        else:
            price_rec_dic = price_recommendations(product_name_list, min_price, max_price)
        #########
        
        if len(purchase_list) == 1:
            purchase_product_name = behavior["product_name"].loc[purchase_list[0]]
        elif len(purchase_list) == 0:
            purchase_product_name = None
        else:
            purchase_product_name = behavior["product_name"].loc[purchase_list[-1]]
            
        if len(view_list) == 1:
            view_product_name = behavior["product_name"].loc[view_list[0]]
        elif len(view_list) == 0 or len(view_list) == 1 and len(purchase_list) == 1:
            view_product_name = None
        else:
            view_product_name = behavior["product_name"].loc[view_list[-1]]
            
        
        if purchase_product_name != None:
            purchase_product_rec = product_recommendations(purchase_product_name) 
        else:
            purchase_product_rec = ""
        if view_product_name == purchase_product_name:
            view_product_name = behavior["product_name"].loc[view_list[-3]]
        view_product_rec = product_recommendations(view_product_name)     

        return render_template("dashboard.html", user=dashboard.user, product=product, purchase_product_rec=purchase_product_rec, view_product_rec=view_product_rec, behavior=behavior, 
                               purchase_product_name=purchase_product_name, view_product_name=view_product_name, price_rec_dic = price_rec_dic )


@app.route('/<product_name>', endpoint='product_rec', methods=['GET','POST'])
@login_required
def product_rec(product_name, cosine_sim=cosine_sim):
    global behavior
    try:
        if login.user is not None:
            product_id = 0
            # Obtain the index of the product
            for i in product.loc[product["title"] == product_name]["product_id"].values:
                product_id = i
            location = product_id -1
            # Get the index of the product that matches the product name
            idx = indices[product_name]

            # Get the pairwise similarity scores
            # Enumerate adds a counter to the iterable and lets it be converted into a list of tuples
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the products based on the similarity scores
            # Reverse gives us the similarity scores in descending order
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the scores of the 10 most similar products
            sim_scores1 = sim_scores[1:11]

            # Get the product indices
            product_indices = [i[0] for i in sim_scores1]

            # Return the top 10 most similar products
            # iloc allows us to retrieve rows from a data frame
            for i in product_indices:
                feature = product[["title","price","image"]].loc[product_indices].values.tolist()
            dictionary = dict(zip(product_indices, feature))
            # Add behavior into behavior.csv
            if product_name != "":
                method = "view"
                new_behavior = {"username": login.user.username, "product_name": product_name, "behavior": method}
                behavior = behavior.append(new_behavior, ignore_index=True)
                if request.form.get('buy'):
                    method = 'buy'
                    new_behavior = {"username": login.user.username, "product_name": product_name, "behavior": method, "price": product["price"].loc[sim_scores[0][0]].tolist()}
                    behavior = behavior.append(new_behavior, ignore_index=True)
            behavior.to_csv('behavior.csv', index = False, encoding='utf-8')
            return render_template("product.html", dictionary=dictionary,location=location, product=product)
    
    except ValueError:
        return "invalid input"
    
@app.route('/shoplist', endpoint='shoplist', methods=['GET','POST'])
def shoplist():
    pass


@app.route('/logout', endpoint='logout', methods=['GET','POST'])
@login_required
def logout():
    logout_user()
    session.pop('username', None)
    return redirect(url_for('login') )


@app.route('/register', endpoint='register', methods=['GET','POST'])
def register():
    form = RegisterForm()
    # When the form validates
    if form.validate_on_submit():
        # A hashed version of the password is created
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        # A new user is created with the hashed password
        new_user = User(username=form.username.data, password=hashed_password)
        # Adds changes to database
        db.session.add(new_user)
        db.session.commit()
        # Redirects to login page
        return redirect(url_for('login'))
    return render_template("register.html", form=form)

if __name__ == '__main__':
    app.run(debug=True)