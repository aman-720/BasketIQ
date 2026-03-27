"""
BasketIQ - Synthetic Data Generator
=====================================
Generates a realistic Instacart-like dataset that exactly mirrors the
real Instacart Market Basket Analysis schema from Kaggle.

USE THIS SCRIPT IF:
  - You don't want to create a Kaggle account, OR
  - You want to reproduce results without downloading ~200MB of data

USE THE REAL DATASET IF:
  - You want authentic transaction patterns for production/research
  - Instructions: See README.md → "Getting the Real Data" section

Output: 5 CSV files saved to data/raw/
  - orders.csv          (~344K rows)
  - order_products.csv  (~2.7M rows)
  - products.csv        (~769 rows)
  - aisles.csv          (134 rows)
  - departments.csv     (21 rows)

Runtime: ~10 seconds
"""
import pandas as pd
import numpy as np
import os, time

np.random.seed(42)
start = time.time()

# Output path is relative to this script's location
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(OUT, exist_ok=True)

# ─── DEPARTMENTS ─────────────────────────────────────────────────────
departments = [
    (1,'frozen'), (2,'other'), (3,'bakery'), (4,'produce'), (5,'alcohol'),
    (6,'international'), (7,'beverages'), (8,'pets'), (9,'dry goods pasta'),
    (10,'bulk'), (11,'personal care'), (12,'meat seafood'), (13,'pantry'),
    (14,'breakfast'), (15,'canned goods'), (16,'dairy eggs'), (17,'household'),
    (18,'babies'), (19,'snacks'), (20,'deli'), (21,'missing')
]
df_dept = pd.DataFrame(departments, columns=['department_id','department'])

# ─── AISLES ──────────────────────────────────────────────────────────
aisles_data = [
    (1,'prepared soups salads',4),(2,'specialty cheeses',16),(3,'energy granola bars',19),
    (4,'instant foods',9),(5,'marinades meat preparation',13),(6,'other',2),
    (7,'packaged meat',12),(8,'bakery desserts',3),(9,'pasta sauce',9),
    (10,'kitchen supplies',17),(11,'cold flu allergy',11),(12,'fresh pasta',9),
    (13,'prepared meals',20),(14,'tofu meat alternatives',4),(15,'packaged seafood',12),
    (16,'fresh fruits',4),(17,'fresh vegetables',4),(18,'whole body lotion',11),
    (19,'baby accessories',18),(20,'candy chocolate',19),(21,'refrigerated',16),
    (22,'lunch meat',20),(23,'cereal',14),(24,'fresh herbs',4),
    (25,'yogurt',16),(26,'popcorn chips',19),(27,'red wines',5),
    (28,'butter',16),(29,'ice cream ice',1),(30,'hot dogs bacon sausage',12),
    (31,'bread',3),(32,'tortillas flat bread',3),(33,'trail mix snack mix',19),
    (34,'missing',21),(35,'frozen meals',1),(36,'cream',16),
    (37,'soft drinks',7),(38,'juice nectars',7),(39,'frozen breakfast',1),
    (40,'oral hygiene',11),(41,'baby bath body care',18),(42,'facial care',11),
    (43,'dish detergents',17),(44,'body lotions soap',11),(45,'condiments',13),
    (46,'deodorants',11),(47,'frozen produce',1),(48,'cleaning products',17),
    (49,'dog food care',8),(50,'shaving needs',11),(51,'laundry',17),
    (52,'soap',11),(53,'coffee',7),(54,'beers coolers',5),
    (55,'paper goods',17),(56,'protein meal replacements',19),(57,'tea',7),
    (58,'preserved dips spreads',13),(59,'frozen pizza',1),(60,'white wines',5),
    (61,'fruit vegetable snacks',19),(62,'first aid',11),(63,'muscles joints pain relief',11),
    (64,'eye ear care',11),(65,'digestion',11),(66,'baby food formula',18),
    (67,'baking ingredients',13),(68,'hair care',11),(69,'canned meals beans',15),
    (70,'energy sports drinks',7),(71,'chips pretzels',19),(72,'frozen appetizers sides',1),
    (73,'canned jarred vegetables',15),(74,'vitamins supplements',11),(75,'nuts seeds dried fruit',19),
    (76,'packaged poultry',12),(77,'air fresheners candles',17),(78,'spices seasonings',13),
    (79,'dips hummus',20),(80,'soup broth bouillon',15),(81,'cat food care',8),
    (82,'rice dried goods',9),(83,'oils vinegars',13),(84,'water seltzer sparkling water',7),
    (85,'specialty wines champagnes',5),(86,'canned fruit applesauce',15),
    (87,'soy lactosefree',16),(88,'packaged vegetables fruits',4),(89,'crackers',19),
    (90,'frozen vegan vegetarian',1),(91,'grains rice oatmeal',14),(92,'buns rolls',3),
    (93,'honeys syrups nectars',13),(94,'plates bowls cups flatware',17),
    (95,'granola',14),(96,'spreads',13),(97,'trash bags liners',17),
    (98,'cocoa drink mixes',7),(99,'dish disposables',17),(100,'mint gum',19),
    (101,'skin care',11),(102,'frozen dessert',1),(103,'feminine care',11),
    (104,'packaged produce',4),(105,'cookies cakes',19),(106,'spirits',5),
    (107,'muscles joints pain',11),(108,'food storage',17),(109,'refrigerated pudding desserts',16),
    (110,'frozen meat seafood',1),(111,'more household',17),(112,'bulk grains rice dried goods',10),
    (113,'bulk dried fruits vegetables',10),(114,'indian foods',6),(115,'latin foods',6),
    (116,'kosher foods',6),(117,'asian foods',6),(118,'frozen juice',1),
    (119,'not available',2),(120,'beauty',11),(121,'diapers wipes',18),
    (122,'packaged cheese',16),(123,'frozen breads doughs',1),(124,'eye care',11),
    (125,'fresh dips tapenades',20),(126,'stomach & digestive',11),
    (127,'cotton balls rounds swabs',11),(128,'missing',21),
    (129,'frozen canned juice',1),(130,'breakfast bakery',14),(131,'cold relief',11),
    (132,'hemp foods',6),(133,'mint candies',19),(134,'ethnic foods',6)
]
df_aisles = pd.DataFrame(aisles_data, columns=['aisle_id','aisle','department_id'])

# ─── PRODUCTS ────────────────────────────────────────────────────────
# Real Instacart product names mapped to their aisles
product_templates = {
    16: ['Banana','Bag of Organic Bananas','Organic Strawberries','Large Lemon','Organic Baby Spinach',
         'Organic Hass Avocado','Organic Avocado','Strawberries','Limes','Organic Raspberries',
         'Organic Blueberries','Cucumber Kirby','Organic Whole Milk','Asparagus','Organic Garlic',
         'Yellow Onions','Organic Zucchini','Organic Lemon','Honeycrisp Apple','Red Pepper',
         'Gala Apples','Navel Oranges','Green Grapes','Fuji Apples','Organic Celery',
         'Organic Ginger Root','Kiwi','Organic Red Onion','Broccoli Crown','Organic Cilantro'],
    17: ['Seedless Red Grapes','Organic Grape Tomatoes','Roma Tomato','Green Bell Pepper',
         'Russet Potato','Sweet Potato','Organic Yellow Onion','Red Onion','Zucchini Squash',
         'English Cucumber','Organic Fuji Apple','Beefsteak Tomato','Fresh Cauliflower',
         'Organic Kale','Organic Cremini Mushrooms','Organic Green Beans','Brussels Sprouts',
         'Iceberg Lettuce','Baby Arugula','Organic Spring Mix','Mixed Greens','Romaine Hearts'],
    25: ['Greek Yogurt','Organic Whole Milk Yogurt','Vanilla Greek Yogurt','Strawberry Yogurt',
         'Blueberry Greek Yogurt','Plain Yogurt','Coconut Yogurt','Peach Yogurt'],
    84: ['Sparkling Water','Spring Water Gallon','Purified Water','Coconut Water',
         'Flavored Sparkling Water','Mineral Water','Electrolyte Water'],
    37: ['Cola','Diet Cola','Ginger Ale','Lemon Lime Soda','Root Beer','Orange Soda','Club Soda'],
    31: ['Whole Wheat Bread','White Bread','Sourdough Bread','Multigrain Bread','Rye Bread',
         'Ciabatta','French Baguette','Pumpernickel Bread'],
    29: ['Vanilla Ice Cream','Chocolate Ice Cream','Strawberry Ice Cream','Cookie Dough Ice Cream',
         'Mint Chocolate Chip','Cookies & Cream','Pistachio Ice Cream'],
    53: ['Medium Roast Coffee','Dark Roast Coffee','French Roast','Organic Coffee','Cold Brew',
         'Espresso Beans','Decaf Coffee','Colombian Coffee'],
    23: ['Honey Nut Cheerios','Frosted Flakes','Granola','Organic Oats','Corn Flakes',
         'Raisin Bran','Cinnamon Toast Crunch','Special K'],
    71: ['Tortilla Chips','Potato Chips','Kettle Chips','Pretzels','Veggie Chips',
         'Sea Salt Chips','BBQ Chips','Sour Cream & Onion Chips'],
    28: ['Unsalted Butter','Salted Butter','Organic Butter','European Butter','Ghee'],
    30: ['Bacon','Turkey Bacon','Chicken Sausage','Pork Sausage','Hot Dogs','Bratwurst'],
    78: ['Black Pepper','Sea Salt','Garlic Powder','Cumin','Paprika','Italian Seasoning',
         'Chili Powder','Oregano','Cinnamon','Turmeric'],
    45: ['Ketchup','Yellow Mustard','Dijon Mustard','Mayonnaise','Hot Sauce','Soy Sauce',
         'Worcestershire Sauce','BBQ Sauce','Ranch Dressing','Italian Dressing'],
    21: ['Organic Whole Milk','2% Reduced Fat Milk','Almond Milk','Oat Milk','Skim Milk',
         'Half & Half','Heavy Cream','Soy Milk','Coconut Milk'],
    38: ['Orange Juice','Apple Juice','Cranberry Juice','Grape Juice','Lemonade','Grapefruit Juice'],
    105: ['Chocolate Chip Cookies','Oreos','Brownie Mix','Birthday Cake','Oatmeal Cookies',
          'Fig Bars','Shortbread Cookies','Animal Crackers'],
    75: ['Almonds','Cashews','Mixed Nuts','Peanuts','Walnuts','Trail Mix','Dried Cranberries',
         'Sunflower Seeds','Pistachios'],
    122: ['Shredded Mozzarella','Cheddar Cheese Block','Cream Cheese','Parmesan','Gouda',
          'Swiss Cheese','Feta Cheese','Brie','Provolone'],
    9:  ['Marinara Sauce','Alfredo Sauce','Pesto','Tomato Basil Sauce','Vodka Sauce','Bolognese'],
    82: ['White Rice','Brown Rice','Jasmine Rice','Basmati Rice','Quinoa','Couscous'],
}

products = []
pid = 1
for aisle_id, names in product_templates.items():
    dept_id = df_aisles[df_aisles.aisle_id == aisle_id].department_id.values[0]
    for name in names:
        products.append((pid, name, aisle_id, dept_id))
        pid += 1

for _, row in df_aisles.iterrows():
    aid = row['aisle_id']
    if aid not in product_templates:
        dept_id = row['department_id']
        aisle_name = row['aisle'].title()
        for j in range(np.random.randint(3, 8)):
            products.append((pid, f"{aisle_name} Item {j+1}", aid, dept_id))
            pid += 1

df_products = pd.DataFrame(products, columns=['product_id','product_name','aisle_id','department_id'])
print(f"[{time.time()-start:.1f}s] Schema ready: {len(df_dept)} depts | {len(df_aisles)} aisles | {len(df_products)} products")

# ─── USERS & ORDERS ──────────────────────────────────────────────────
N_USERS  = 11_000
N_ORDERS = 350_000

user_ids = np.arange(1, N_USERS + 1)
orders_per_user = np.random.negative_binomial(3, 0.08, size=N_USERS)
orders_per_user = np.clip(orders_per_user, 3, 100)
scale = N_ORDERS / orders_per_user.sum()
orders_per_user = np.maximum(3, (orders_per_user * scale).astype(int))

orders = []
oid = 1
for uid in user_ids:
    n_orders = orders_per_user[uid - 1]
    for order_num in range(1, n_orders + 1):
        dow       = np.random.choice(range(7), p=[0.18,0.16,0.12,0.11,0.12,0.14,0.17])
        hour      = int(np.clip(np.random.normal(13, 4), 0, 23))
        days_since = (np.nan if order_num == 1
                      else np.random.choice([7,14,21,30,3,5,10,6,8,4,15,28,1,2],
                                            p=[0.206,0.155,0.103,0.124,0.052,0.052,0.062,0.052,0.041,
                                               0.031,0.041,0.052,0.015,0.010]))
        eval_set  = 'prior' if order_num < n_orders else np.random.choice(['train','test'])
        orders.append((oid, uid, eval_set, order_num, dow, hour, days_since))
        oid += 1

df_orders = pd.DataFrame(orders, columns=['order_id','user_id','eval_set','order_number',
                                           'order_dow','order_hour_of_day','days_since_prior_order'])
print(f"[{time.time()-start:.1f}s] Orders generated: {len(df_orders):,}")

# ─── ORDER_PRODUCTS (~2.7M rows) ─────────────────────────────────────
n_products  = len(df_products)
product_ids = df_products.product_id.values

# Zipf-like popularity weights, boosted for fresh produce & dairy
ranks   = np.arange(1, n_products + 1)
weights = 1.0 / (ranks ** 0.8)
for i, (_, row) in enumerate(df_products.iterrows()):
    if row['aisle_id'] in [16, 17, 25, 84, 21, 24]:
        weights[i] *= 3.0
    elif row['aisle_id'] in [31, 53, 71, 122, 29]:
        weights[i] *= 2.0
weights /= weights.sum()

all_order_ids    = df_orders.order_id.values
items_per_order  = np.clip(np.random.normal(9, 4, size=len(all_order_ids)).astype(int), 1, 40)
total_items      = items_per_order.sum()
print(f"[{time.time()-start:.1f}s] Generating {total_items:,} item rows...")

order_id_rep = np.repeat(all_order_ids, items_per_order)
cart_order   = np.concatenate([np.arange(1, n+1) for n in items_per_order])
selected     = np.random.choice(product_ids, size=total_items, p=weights)
reordered    = np.random.binomial(1, 0.6, size=total_items)

df_op = pd.DataFrame({'order_id': order_id_rep, 'product_id': selected,
                       'add_to_cart_order': cart_order, 'reordered': reordered})
df_op = df_op.drop_duplicates(subset=['order_id','product_id'], keep='first')
df_op['add_to_cart_order'] = df_op.groupby('order_id').cumcount() + 1

# ─── SAVE ─────────────────────────────────────────────────────────────
df_dept.to_csv(f"{OUT}/departments.csv", index=False)
df_aisles[['aisle_id','aisle']].to_csv(f"{OUT}/aisles.csv", index=False)
df_products.to_csv(f"{OUT}/products.csv", index=False)
df_orders.to_csv(f"{OUT}/orders.csv", index=False)
df_op.to_csv(f"{OUT}/order_products.csv", index=False)

elapsed = time.time() - start
print(f"\n{'='*55}")
print(f"  Synthetic dataset generated in {elapsed:.1f}s")
print(f"{'='*55}")
print(f"  Users:          {N_USERS:,}")
print(f"  Orders:         {len(df_orders):,}")
print(f"  Products:       {len(df_products):,}")
print(f"  Transactions:   {len(df_op):,}")
print(f"  Saved to:       {OUT}/")
print(f"{'='*55}")
