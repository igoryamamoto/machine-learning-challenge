## to view specific line: sed "${LINE_NUM}q;d"

file_name=$1

# Begining of products pageview events
# {"page_type": "product", "productId": "f53722f6438544e81d68e6adcf8609817ea50728", "timestamp": "2016-01-05 20:17:56", "source": NaN, "gender": "M", "uid": "902ba3cda1883801594b6e1b452790cc53948fda", "event_type": "pageview"}

products=$(grep productId -m 1 -n $file_name | cut -f 1 -d :)

# Begining of products purchase
# {"date": "2017-02-14 19:01:07", "products": [{"pid": "0d5597cab610822b4788e2ada9b83cddb9639f75", "quantity": 1.0}], "source": "desktop", "gender": "F", "uid": "29ef06c0a1b63e87b217e59ca967deed258c600a", "event_type": "purchase"}

purchase=$(grep purchase -m 1 -n $file_name | cut -f 1 -d :)

total=$(wc -l $file_name | cut -f 1 -d ' ')

# Split data file 
split -l $(($products - 1)) $file_name data_ -a 1

mv data_a pageviews_$file_name

T=$(($purchase - $products))
B=$(($total - $purchase))

# create the top of file: 
head -n $T data_b > products_$file_name

# create bottom of file: 
tail -n $B data_b > purchase_$file_name

rm data_b
