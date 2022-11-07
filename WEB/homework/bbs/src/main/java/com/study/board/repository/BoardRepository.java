package com.study.board.repository;

import com.study.board.entity.Board;
<<<<<<< .merge_file_J5I7b7
=======
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
>>>>>>> .merge_file_vGiwb0
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository                                           //엔티티, 기본키 타입?
public interface BoardRepository extends JpaRepository<Board,Integer> {
<<<<<<< .merge_file_J5I7b7
=======
//   find(컬럼이름)Containing -> 컬럼이름을 정할 때 쓴것처럼 단어에 따라 앞은 대문자로 해줘야한다.
    Page<Board> findByTitleContaining(String searchKeyword, Pageable pageable);
>>>>>>> .merge_file_vGiwb0
}
