package com.study.board.repository;

import com.study.board.entity.Board;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository                                           //엔티티, 기본키 타입?
public interface BoardRepository extends JpaRepository<Board,Integer> {
}