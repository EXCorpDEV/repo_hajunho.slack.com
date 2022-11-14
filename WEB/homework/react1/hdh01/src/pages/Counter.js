import React, {useState} from 'react';

const Counter = () => {
  const [num, setNumber] = useState(0);

  const increase = () => {
    setNumber(num + 1); // 버튼을 두개를 만들어서 +버튼을 누르면 1씩증가 -버튼을 누르면 1씩 감소
  };

  const decrease = () => {
    setNumber(num - 1);
  };

  return (
    <div>
      <button onClick={increase}>+1</button> 
      <button onClick={decrease}>-1</button>
      <p>{num}</p>
    </div>
  );
};

export default Counter;