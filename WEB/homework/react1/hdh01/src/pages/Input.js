import React, {useState} from 'react';

const Input = () => {
  const [txtValue, setTextValue] = useState("");

  const onChange = (e) => {
    setTextValue(e.target.value) // input type = text를 의미 거기의 value를 체인지이벤트가 일어날때마다 setTextValue에 넣어주는것
  };

  return (
    <div>
      <input type="text" value={txtValue} onChange={onChange} />
      <br />
      <p>{txtValue}</p>
    </div>
  );
};

export default Input;