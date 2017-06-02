file_name=$1

mkdir sub_$file_name
grep productId $file_name > sub_$file_name/products
grep purchase $file_name > sub_$file_name/purchase
grep cart $file_name > sub_$file_name/cart
grep checkout $file_name > sub_$file_name/checkout
grep confirmation $file_name > sub_$file_name/confirmation
grep home $file_name > sub_$file_name/home
grep \"category $file_name > sub_$file_name/category
grep search $file_name > sub_$file_name/search
grep subcategory $file_name > sub_$file_name/subcategory
grep brand_landing $file_name > sub_$file_name/brand_landing

