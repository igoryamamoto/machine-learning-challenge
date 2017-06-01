# Get file name to be splitted as argument
file_name=$1

# Begining of products pageview events
products=$(grep productId -m 1 -n $file_name | cut -f 1 -d :)

# Begining of products purchase
purchase=$(grep purchase -m 1 -n $file_name | cut -f 1 -d :)

# Total number of lines
total=$(wc -l $file_name | cut -f 1 -d ' ')

# Split data file (it works if the top part is larger than the bottom, otherwise it'll create a lot of subfiles) 
split -l $(($products - 1)) $file_name split_ -a 1

mv split_a pageviews_$file_name

T=$(($purchase - $products))
B=$(($total - $purchase))

# Create products file
head -n $T split_b > products_$file_name

# Create purchase file
tail -n $B split_b > purchase_$file_name

rm split_b
