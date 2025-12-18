import streamlit as st

# adding title 
st.title("My first streamlit project 111")

# Adding simple text
st.write("Here is the text using the st.write command")

# Adding slider
number = st.slider("Here is the slider", 1,100,10)
st.write("the number is :", number)


# Adding a button
if st.button('Say Hello'):
    st.write("Why hello there")
else:
    st.write("Good Bye")    

# Adding the radio with options

genra = st.radio(" How are you? ", ('Good', 'Nice', 'Excellent'))    

st.write("the genra is:", genra)

# adding the selectbox option dropdown list

option = st.selectbox("what was the temperature of your area  ",('low','medium', 'high'))

st.write('the answer is :', option)

option_slider = st.sidebar.selectbox("what was the temperature of your area  ",('low','medium', 'high'))

st.write('the answer is :', option_slider)


# adding the test
st.sidebar.text_input('enter the number')